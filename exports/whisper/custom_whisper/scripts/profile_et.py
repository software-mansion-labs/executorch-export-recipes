import argparse
import ast
import ctypes
import os
import pandas as pd
import sys
import tabulate
import torch
from colorama import Fore, Style, init as colorama_init
from executorch.devtools import Inspector
from executorch.runtime import Runtime, Verification

# Link dynamic libraries to support LLM extension
dylib_paths = [
  os.path.join(
    sys.prefix,
    "lib",
    "python3.12",
    "site-packages",
    "executorch",
    "extension",
    "llm",
    "custom_ops",
    "libcustom_ops_aot_lib.dylib"
  ),
  os.path.join(
    sys.prefix,
    "lib",
    "python3.12",
    "site-packages",
    "executorch",
    "kernels",
    "quantized",
    "libquantized_ops_aot_lib.dylib"
  ),
]
for path in dylib_paths:
  if os.path.exists(path):
      ctypes.cdll.LoadLibrary(path)
  else:
      print(f"Warning: {path} not found. LLM extension may not work.")


if __name__ == "__main__":
    
  runtime = Runtime.get()

  colorama_init(autoreset=True)

  # Define script arguments
  parser = argparse.ArgumentParser(description="Profiling script")
  parser.add_argument('--model', type=str, required=True, help='Path to input .pte file')
  parser.add_argument('--model-input', type=str, help='Path to input tensors .pt file')
  parser.add_argument("--info", action="store_true", help="Enable info mode")
  parser.add_argument('--output', type=str, default="", help='Path to output artifacts directory')
  parser.add_argument('--n-tests', type=int, default=1, help='Number of tests to run')

  args = parser.parse_args()

  # ----------------------------------
  # STEP 1 - load the executorch model
  # ----------------------------------

  model_filepath = args.model
  
  try:
    executorch_program = runtime.load_program(
      model_filepath, 
      verification=Verification.Minimal,
      enable_etdump=True
    )
    # forward_method = executorch_program.load_method("forward_512")
    # forward_method = executorch_program.load_method("encode")
    forward_method = executorch_program.load_method("decode")
    
  except FileNotFoundError:
    print(f"Error: unable to open file {model_filepath}. Make sure that the file exists.")
    sys.exit(1)
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
  
  print(f"Succesfully loaded model from {model_filepath}")

  if args.info:
    print(Fore.CYAN + Style.BRIGHT + "Model Metadata:" + Style.RESET_ALL)
    print(Fore.CYAN + str(forward_method.metadata) + Style.RESET_ALL)

    sys.exit(0)
  
  # -----------------------------------
  # STEP 2 - generate the input tensors
  # -----------------------------------

  # User-defined input generator
  def generate_inputs():
    # <---------------------->
    # DEFINE MODEL INPUTS HERE
    # <---------------------->
    return (
      # torch.randn(1, 3, 512, 512, dtype=torch.float32),
      # torch.randn(480000, dtype=torch.float32)
      torch.randint(1, 100, size=(1, 1)),
      torch.arange(0, 1, dtype=torch.long),
      torch.randn(1, 1500, 384, dtype=torch.float32),
    )
    raise NotImplementedError("")

  # If user provided an input file or defined an input generator, use the provided tensors
  # Otherwise try to generate random tensors based on the static shapes of the model's method
  if args.model_input:
    inputs_filepath = args.model_input

    try:
      inputs = torch.load(inputs_filepath)
    except FileNotFoundError:
      print(f"Error: unable to open file {inputs_filepath}. Make sure that the file exists.")
      sys.exit(1)
    
    print("Succesfully loaded inputs:", inputs)
  else:
    try:
      inputs = generate_inputs()

      print("Using user-defined inputs:", inputs)
    except NotImplementedError:
      print("Input function not defined - switching to random input generator fallback...")

      no_inputs = forward_method.metadata.num_inputs()

      inputs = ()
      for i in range (no_inputs):
        input_meta = forward_method.metadata.input_tensor_meta(i)
        input_size = input_meta.sizes()
        input_type = input_meta.dtype()

        # TODO: map other types
        type_mappings = {
          4: (torch.long, torch.randint, (1, 100)),
          6: (torch.float32, torch.randn, ()),
          11: (torch.bool, torch.randint, (0, 2)),
        }

        input_dtype, generator, extra_args = type_mappings[input_type]
        sample_input = generator(*extra_args, size=input_size, dtype=input_dtype)

        inputs = inputs + (sample_input,)
      
      print("Succesfully generated random inputs:", inputs)

  # -----------------------------
  # STEP 3 - generate ETDump file
  # -----------------------------

  output_directory = args.output
  etdump_output_path = f"{output_directory}/etdump.etdp"
  debug_file_output_path = f"{output_directory}/debug.bin"

  # Create a debug file if not present
  with open(debug_file_output_path, 'wb') as debug_file:
    pass

  use_existing_etdump = False
  if os.path.exists(etdump_output_path):
    print(Fore.YELLOW + f"File '{etdump_output_path}' already exists. Use existing one? (y/n): " + Style.RESET_ALL, end='', flush=True)
    user_choice = input().strip().lower()
    if user_choice == "y":
      use_existing_etdump = True

  if not use_existing_etdump:
    # Execute the forward method given amount of times and collect inference stats
    for _ in range(args.n_tests):
      forward_method.execute(inputs)

    # Create ETDump artifact file
    executorch_program.write_etdump_result_to_file(etdump_output_path, debug_file_output_path)

    print("Succesfully generated and saved ETDump file as", etdump_output_path)
  else:
    print("Using existing ETDump file:", etdump_output_path)

  # ------------------------------------
  # STEP 4 - profile and display results
  # ------------------------------------

  def _to_list(v):
    # Normalize 'raw' values to a list[float]
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple)):
                return [float(x) for x in parsed]
            return [float(parsed)]
        except Exception:
            return []
    if pd.isna(v):
        return []
    try:
        return [float(v)]
    except Exception:
        return []

  def display_results(df: pd.DataFrame) -> None:
      # 1) Per-op total time and share of Method::execute
      exe_df = df[df['event_block_name'] == 'Execute'].copy()
      exe_df['raw_list'] = exe_df['raw'].apply(_to_list)
      exe_df['op_time_ms'] = exe_df['raw_list'].apply(lambda xs: sum(float(x) for x in xs))
      exe_df['call_count'] = exe_df['raw_list'].apply(len)
      exe_df['delegated_calls'] = exe_df.apply(
          lambda r: r['call_count'] if bool(r.get('is_delegated_op', False)) else 0, axis=1
      )

      stats = exe_df.groupby('event_name', as_index=False).agg(
          total_time_ms=('op_time_ms', 'sum'),
          total_calls=('call_count', 'sum'),
          delegated_calls=('delegated_calls', 'sum'),
      )

      method_mask = df['event_name'].astype(str).str.fullmatch(r'Method:{1,2}execute')
      method_rows = df[method_mask]
      total_execute_time_ms = (
          sum(sum(_to_list(v)) for v in method_rows['raw'])
          if not method_rows.empty
          else float(stats['total_time_ms'].sum())
      )

      stats['share_pct'] = (
          stats['total_time_ms'] / total_execute_time_ms * 100.0
      ) if total_execute_time_ms > 0 else 0.0

      # 2) Delegation counts/ratio per operation
      stats['delegated_ratio_pct'] = (
          stats['delegated_calls'] / stats['total_calls'] * 100.0
      ).fillna(0.0)

      stats = stats.sort_values('total_time_ms', ascending=False)

      def color_row(row):
          # Method::execute (and variants), DELEGATE_CALL, OPERATOR_CALL should remain uncolored
          event_name = str(row['event_name'])
          is_method_execute = event_name.startswith("Method:") or event_name.startswith("Method::")
          is_special = event_name in ("DELEGATE_CALL", "OPERATOR_CALL")
          base = ""
          if is_method_execute or is_special:
              base = Style.BRIGHT  # white/bold, no color
          elif row['delegated_ratio_pct'] == 0:
              base = Fore.RED + Style.BRIGHT
          elif row['delegated_ratio_pct'] == 100:
              base = Fore.GREEN + Style.BRIGHT
          elif 0 < row['delegated_ratio_pct'] < 100:
              base = Fore.YELLOW + Style.BRIGHT
          else:
              base = ""
          # Top 3 by time: keep bold, but don't override Method::execute or special coloring
          if not (is_method_execute or is_special):
              if row['name'] == 0:
                  base += Style.BRIGHT
              elif row['name'] == 1:
                  base += Style.BRIGHT
              elif row['name'] == 2:
                  base += Style.BRIGHT
          # Remove 'name' from output columns
          values = [
              row['event_name'],
              f"{row['total_time_ms']:.3f}",
              f"{row['share_pct']:.2f}%",
              int(row['total_calls']),
              int(row['delegated_calls']),
              f"{row['delegated_ratio_pct']:.2f}%",
          ]
          return [base + str(x) + Style.RESET_ALL for x in values]

      # Prepare table data (kept if you need non-colored data elsewhere)
      table_data = []
      for idx, row in stats.iterrows():
          formatted = [
              row['event_name'],
              f"{row['total_time_ms']:.3f}",
              f"{row['share_pct']:.2f}%",
              int(row['total_calls']),
              int(row['delegated_calls']),
              f"{row['delegated_ratio_pct']:.2f}%",
          ]
          table_data.append(formatted)

      # Add header styling
      headers = [
          Fore.BLUE + Style.BRIGHT + "Op Name" + Style.RESET_ALL,
          Fore.BLUE + Style.BRIGHT + "Total Time (ms)" + Style.RESET_ALL,
          Fore.BLUE + Style.BRIGHT + "Share (%)" + Style.RESET_ALL,
          Fore.BLUE + Style.BRIGHT + "Calls" + Style.RESET_ALL,
          Fore.BLUE + Style.BRIGHT + "Delegated" + Style.RESET_ALL,
          Fore.BLUE + Style.BRIGHT + "Delegated (%)" + Style.RESET_ALL,
      ]

      # Color rows
      styled_table = []
      for i, row in enumerate(stats.itertuples(index=False)):
          row_dict = row._asdict()
          row_dict['name'] = i
          styled_table.append(color_row(row_dict))

      print(Fore.YELLOW + Style.BRIGHT + "\nPer-op Execute stats (ms and delegation):\n" + Style.RESET_ALL)
      print(
          tabulate.tabulate(
              styled_table,
              headers=headers,
              tablefmt="fancy_grid",
              stralign="right",
              numalign="right",
          )
      )

  print("Profiling with Inspector...")

  inspector = Inspector(etdump_path=etdump_output_path, debug_buffer_path=debug_file_output_path)
  tabular_data = inspector.to_dataframe()

  print("Profiling finished. Generating profile stats...")

  display_results(tabular_data)