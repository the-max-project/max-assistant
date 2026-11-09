from collections import defaultdict
import csv
import json
import re
import sys


def extract_base_test_name(testcase_name):
  """Strips pytest wrappers and model-specific prefixes to isolate the test case name."""
  match = re.search(r"\[(?:.*?-)?([a-zA-Z0-9_]+-[a-zA-Z0-9_]+)\]$", testcase_name)
  if match:
    return match.group(1)
  if "[" in testcase_name and testcase_name.endswith("]"):
    raw = testcase_name[testcase_name.find("[") + 1 : -1]
    parts = raw.split("-", 1)
    return parts[1] if len(parts) > 1 else raw
  return testcase_name


def process_benchmark_jsonl(jsonl_file_path, pivot_csv_path, flat_csv_path):
  records = []
  with open(jsonl_file_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        records.append(json.loads(line))

  # ---------------------------------------------------------
  # 1. Flat CSV: Model, Test Case, Step, Prompt, Elapsed Time
  # ---------------------------------------------------------
  flat_headers = ["Model", "Test Case", "Step", "Prompt", "Elapsed Time (s)"]

  with open(flat_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(flat_headers)

    for r in records:
      model = r.get("model", "")
      raw_test = r.get("testcase_name", "")
      test_name = extract_base_test_name(raw_test)
      step_results = r.get("step_results", [])

      if step_results:
        for step_info in step_results:
          writer.writerow([
              model,
              test_name,
              step_info.get("step", ""),
              step_info.get("user_input", ""),
              round(step_info.get("elapsed_time", 0.0), 4),
          ])
      else:
        # Test failed or exited before executing prompt steps
        writer.writerow([model, test_name, "", "", ""])

  # ---------------------------------------------------------
  # 2. Pivoted CSV: Models as columns with Total & Avg rows
  # ---------------------------------------------------------
  models = []
  unique_tests = []
  seen_tests = set()

  test_totals = defaultdict(dict)
  prompt_times = defaultdict(dict)

  step_keys = []
  seen_step_keys = set()

  for r in records:
    model = r.get("model", "")
    if model and model not in models:
      models.append(model)

    raw_test = r.get("testcase_name", "")
    test_name = extract_base_test_name(raw_test)

    if test_name not in seen_tests:
      seen_tests.add(test_name)
      unique_tests.append(test_name)

    total_time = r.get("total_execution_time", 0.0)
    test_totals[test_name][model] = total_time

    for step_info in r.get("step_results", []):
      step_num = step_info.get("step", 1)
      prompt_text = step_info.get("user_input", "")
      key = (test_name, step_num, prompt_text)

      if key not in seen_step_keys:
        seen_step_keys.add(key)
        step_keys.append(key)

      prompt_times[(test_name, step_num)][model] = step_info.get(
          "elapsed_time", 0.0
      )

  test_to_steps = defaultdict(list)
  for test_name, step_num, prompt_text in step_keys:
    test_to_steps[test_name].append((step_num, prompt_text))

  pivot_headers = ["Test Case", "Step", "Prompt (User Input)"] + models

  with open(pivot_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(pivot_headers)

    for test in unique_tests:
      steps = test_to_steps.get(test, [])
      for step_num, prompt_text in steps:
        row = [test, step_num, prompt_text]
        for m in models:
          val = prompt_times.get((test, step_num), {}).get(m, "")
          row.append(f"{val:.4f}" if isinstance(val, (int, float)) else "")
        writer.writerow(row)

      # Insert row for Test Total Time
      total_row = [test, "", "TOTAL TEST TIME (s)"]
      for m in models:
        t_val = test_totals.get(test, {}).get(m, "")
        total_row.append(
            f"{t_val:.4f}" if isinstance(t_val, (int, float)) else ""
        )
      writer.writerow(total_row)

    # Insert overall row for Model Average Test Time
    avg_row = ["ALL TESTS", "", "AVERAGE TEST TIME (s)"]
    for m in models:
      times = [test_totals[t][m] for t in unique_tests if m in test_totals[t]]
      avg_time = sum(times) / len(times) if times else 0.0
      avg_row.append(f"{avg_time:.4f}")
    writer.writerow(avg_row)


if __name__ == "__main__":
  input_file = (
      sys.argv[1]
      if len(sys.argv) > 1
      else "RTX-5090-Run2-20261106_230501.jsonl"
  )
  pivot_file = (
      sys.argv[2] if len(sys.argv) > 2 else "benchmark_pivot_summary.csv"
  )
  flat_file = sys.argv[3] if len(sys.argv) > 3 else "benchmark_flat_summary.csv"

  process_benchmark_jsonl(input_file, pivot_file, flat_file)
  print(f"Generated pivoted CSV: {pivot_file}")
  print(f"Generated flat CSV:    {flat_file}")