# %env KERAS_BACKEND=torch
import torch
from model_keras import load_model, RotaryEmbedding, Arange
from transcribe import load_audio, transcribe_pt, transcribe
from model_pt import MoonshineModel
import moonshine
import keras
import lovely_tensors as lt
lt.monkey_patch()
from pathlib import Path

ASSETS_DIR = Path(__file__).parents[0] / "assets"


def convert_keras_to_pytorch(model_name='moonshine/tiny', device='cpu', verify=True):
    """
    Convert a Keras Moonshine model to PyTorch.

    Args:
        model_name: Name of the Keras model to load (default: 'moonshine/tiny')
        device: Device to load the model on (default: 'cpu')
        verify: Whether to verify the conversion with assertions (default: True)

    Returns:
        MoonshineModel: Converted PyTorch model
    """
    # Initialize PyTorch model
    pt_model = MoonshineModel(288, 288, 8, 6, 6, dec_ff_swiglu=True)
    pt_model.to(device)

    # Load Keras model
    keras_model = load_model(model_name)

    # Convert preprocessor parameters
    preprocessor_params = [x.value.permute((*range(-1,-len(x.value.shape)-1,-1),)) for x in keras_model.preprocessor.preprocess.weights]
    for value, new_value in zip(pt_model.preprocessor.state_dict().values(), preprocessor_params):
        value.copy_(new_value.contiguous())

    if verify:
        # Verify preprocessor conversion
        input = load_audio(moonshine.ASSETS_DIR / 'beckett.wav')
        encoder_input = keras_model.preprocessor(input)
        print(keras_model.preprocessor(input).data)
        print(encoder_input.data)
        assert(torch.allclose(keras_model.preprocessor(input).data, encoder_input.data, atol=1e-5))

    # Convert encoder parameters
    encoder_params = [
            x.value.permute((*range(-1,-len(x.value.shape)-1,-1),))
            if len(x.value.size())!=3 else x.value
            for x in keras_model.encoder.encoder.weights
        ][1:]

    encoder_params = [x.contiguous().view(288,288).T if len(x.size())==3 else x for x in encoder_params]
    for value, new_value in zip(pt_model.encoder.state_dict().values(), encoder_params):
        value.copy_(new_value.contiguous())

    if verify:
        # Verify encoder conversion
        print(pt_model.encoder(encoder_input).data)
        print(keras_model.encoder(encoder_input, seq_len=keras.ops.convert_to_tensor([413])).data)
        assert(torch.allclose(
          pt_model.encoder(encoder_input).data,
          keras_model.encoder(encoder_input, seq_len=keras.ops.convert_to_tensor([413])).data,
          atol=1e-5
        ))

    # Convert uncached decoder parameters
    uncached_decoder_params = [
            x.value.permute((*range(-1,-len(x.value.shape)-1,-1),)) if len(x.value.size())!=3 else x.value for x in keras_model.decoder.uncached_call.weights
        ][2:]
    uncached_decoder_params = [x.contiguous().view(288,288).T if len(x.size())==3 else x for x in uncached_decoder_params]
    for value, new_value in zip(pt_model.decoder.state_dict().values(), uncached_decoder_params):
        value.copy_(new_value)

    # Convert cached decoder parameters
    cached_decoder_params = [
            x.value.permute((*range(-1,-len(x.value.shape)-1,-1),)) if len(x.value.size())!=3 else x.value for x in keras_model.decoder.cached_call.weights
        ][2:]
    cached_decoder_params = [x.contiguous().view(288,288).T if len(x.size())==3 else x for x in cached_decoder_params]
    for value, new_value in zip(pt_model.decoder.state_dict().values(), cached_decoder_params):
        value.copy_(new_value)

    # Convert token embedding
    layers = [x.name for x in keras_model.decoder.uncached_call.layers]
    keras_rev_emb = keras.Model(
      inputs=keras_model.decoder.uncached_call.get_layer(layers[3]).input,
      outputs=keras_model.decoder.uncached_call.get_layer(layers[3]).output
    )
    pt_model.decoder.state_dict()['token_embedding.weight'].copy_(keras_rev_emb.weights[0].value)

    if verify:
        # Verify full model conversion
        assert(['Ever tried ever failed, no matter try again, fail again, fail better.'] == transcribe_pt(pt_model, moonshine.ASSETS_DIR / 'beckett.wav') )
        print(transcribe_pt(pt_model, moonshine.ASSETS_DIR / 'beckett.wav'))

    return pt_model


if __name__ == '__main__':
    # Run conversion and save model when script is executed directly
    pt_model = convert_keras_to_pytorch(device='mps', verify=True)
    torch.save(pt_model.state_dict(), './moonshine.pth')
