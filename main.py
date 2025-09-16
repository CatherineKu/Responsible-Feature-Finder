import os
import glob
import time
import pandas as pd
from openai import OpenAI

# Initialize client
client = OpenAI(api_key=os.getenv(""))


# 数据集的根目录
ROOT_DIR = "/Users/chuhanku/Documents/Black Lab/LLM_modelBuilding/ReadytoRun" 
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
RATE_LIMIT_SLEEP = 0.6 
OUTPUT_CSV = "predictive_targets.csv"


def read_first_line(path: str) -> str:
    """Read the frist line of txt"""
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
    return (line or "").strip()

def safe_read_csv(path: str, nrows: int = 5) -> pd.DataFrame:
    """读取CSV前 nrows 行；若不存在或出错则返回空DF"""
    try:
        if os.path.exists(path):
            return pd.read_csv(path).head(nrows)
    except Exception:
        pass
    return pd.DataFrame()

def parse_llm_response(text: str):
    goal = ""
    targets = ""
    rationale_lines = []
    section = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("Goal:"):
            goal = line.replace("Goal:", "", 1).strip()
            section = None
        elif line.startswith("Predictive Target(s):"):
            targets = line.replace("Predictive Target(s):", "", 1).strip()
            section = None
        elif line.startswith("Rationale:"):
            rationale = line.replace("Rationale:", "", 1).strip()
            rationale_lines.append(rationale)
            section = "rationale"
        else:
            # If rationale spans multiple lines
            if section == "rationale":
                rationale_lines.append(line)

    return {
        "goal": goal,
        "targets": targets,
        "rationale": " ".join(rationale_lines)
    }



def create_prompt(dataset_name, description, goal_line, simplified_df):
    return f"""
You are given a dataset summary and a high-level goal. Propose the most appropriate predictive target(s).

Return a STRICT JSON object ONLY, no extra text, matching this schema:
{{
  "dataset_name": "string",
  "goal": "string",
  "predictive_targets": ["string", "..."],  // column names from the dataset
  "rationale": "string"
}}

Constraints:
- predictive_targets must be column names that appear in the dataset.
- Do not invent columns not present in the dataset.
- If multiple columns are reasonable, include all of them in the array.

Inputs:
- Dataset name: {dataset_name}
- Description: {description}
- Policy/Business Goal (first line): {goal_line}
- Simplified dataset (first 5 rows, CSV):
{simplified_df}
"""


def main():
    rows = []
    dataset_count = 0

    # 遍历每个数据集文件夹
    for dataset_name in sorted(os.listdir(ROOT_DIR)):
        ds_dir = os.path.join(ROOT_DIR, dataset_name)
        if not os.path.isdir(ds_dir):
            continue
        
        '''
        if dataset_count >= 5:
            break
        dataset_count += 1
        '''
        # 读取 description.txt
        desc_path = os.path.join(ds_dir, "description.txt")
        description = read_first_line(desc_path) if os.path.exists(desc_path) else ""

        # 读取 simplified.csv（只取前5行）
        simp_path = os.path.join(ds_dir, "simplified.csv")
        simplified_df = safe_read_csv(simp_path, nrows=5)

        # 匹配所有 goal*.txt
        goal_paths = sorted(glob.glob(os.path.join(ds_dir, "goal*.txt")))
        if not goal_paths:
            # 没有 goal 文件也记录一条（可选）
            goal_paths = []

        for goal_path in goal_paths:
            goal_line = read_first_line(goal_path)
            prompt = create_prompt(dataset_name, description, goal_line, simplified_df)

            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"} 
                )
                content = resp.choices[0].message.content.strip()
                data = json.loads(content)
                '''
                print("------")
                print(prompt)
                print("------") 
                '''
            except Exception as e:
                content = f"[ERROR] {type(e).__name__}: {e}"
            
            parsed = parse_llm_response(content)

            rows.append({
                "dataset_name": dataset_name,
                "goal_file": os.path.basename(goal_path) if goal_path else "",
                "goal_first_line": goal_line,
                "goal": parsed["goal"],
                "predictive_targets": parsed["targets"],
                "rationale": parsed["rationale"],
                "llm_response_raw": content
            })

            time.sleep(RATE_LIMIT_SLEEP)
        

    # 保存为 CSV
    out_df = pd.DataFrame(
        rows,
        columns=[
            "dataset_name",
            "goal_file",
            "goal_first_line",
            "goal",
            "predictive_targets",
            "rationale",
            "llm_response_raw",
        ],
    )
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Done. Results saved to {OUTPUT_CSV} (rows={len(out_df)})")

if __name__ == "__main__":
    main()