from collections import OrderedDict

import torch
import torch.nn as nn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import ExportedProgram, export

# latin = 352 - done
# en = 97 - done
# japanes = 2215 - done
# ch_sim = 6719 - done
# korean = 1009 - done
# telugu = 166 - done
# kannada = 168 - done
# cyrillic = 208 - done


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        # try: # multi gpu needs this
        #     self.rnn.flatten_parameters()
        # except: # quantization doesn't work with this
        #     pass
        recurrent, _ = self.rnn(
            input
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(
                self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.Conv2d(
                self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.ConvNet(input)


class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AvgPool = nn.AvgPool2d(kernel_size=(1, 3), stride=1)
        # self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input):
        """Feature extraction stage"""
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AvgPool(visual_feature.permute(0, 3, 1, 2))
        print(visual_feature.shape)
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction


# Language configurations: (language_name, weight_file, num_classes)
languages = [
    ("latin", "latin_g2.pth", 352),
    ("english", "english_g2.pth", 97),
    ("japanese", "japanese_g2.pth", 2215),
    ("zh_sim", "zh_sim_g2.pth", 6719),
    ("korean", "korean_g2.pth", 1009),
    ("telugu", "telugu.pth", 166),
    ("kannada", "kannada.pth", 168),
    ("cyrillic", "cyrillic_g2.pth", 208),
]

# Model input sizes
sizes = [64, 128, 256, 512]

torch.manual_seed(42)

# Export models for each language and size
for lang_name, weight_file, num_classes in languages:
    print(f"\n{'='*60}")
    print(f"Exporting {lang_name} models (num_classes={num_classes})")
    print(f"{'='*60}")

    # Create model with correct num_classes for this language
    model = Model(
        num_class=num_classes, input_channel=1, output_channel=256, hidden_size=256
    )

    # Load weights
    state_dict = torch.load(weight_file, map_location="cpu", weights_only=False)

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:]
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    # Export for each size
    exported_programs: list[ExportedProgram] = []
    for size in sizes:
        sample_inputs = (torch.ones(1, 1, 64, size),)
        exported_programs.append(export(model, sample_inputs))
    print(f"✓ Completed {lang_name}")

    edge = to_edge_transform_and_lower(
        {
            "forward_64": exported_programs[0],
            "forward_128": exported_programs[1],
            "forward_256": exported_programs[2],
            "forward_512": exported_programs[3],
        },
        partitioner=[XnnpackPartitioner()],
    )
    exec_prog = edge.to_executorch()

    output_filename = f"xnnpack_crnn_{lang_name}.pte"
    with open(output_filename, "wb") as file:
        exec_prog.write_to_file(file)
    print(f"  ✓ Saved: {output_filename}")

print(f"\n{'='*60}")
print("All models exported successfully!")
print(f"{'='*60}")
