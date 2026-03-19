"""
AI饭碗系统 - 职业数据生成器
===============================
使用 DeepSeek API 批量生成中国全部职业的AI冲击评分、薪资、就业人数等数据

使用方法:
1. 安装依赖: pip install openai pandas
2. 设置环境变量: export DEEPSEEK_API_KEY="your-api-key"
3. 准备输入文件: occupations_input.csv (至少包含 code, name, major_category, major_category_name)
4. 运行: python generate_occupations.py
5. 输出: occupations_full.csv (包含完整数据，可直接导入AI饭碗系统)

输入CSV格式:
code,name,major_category,major_category_name
1-01-00-00,中国共产党中央委员会和地方各级党组织负责人,1,党的机关、国家机关、群众团体和社会组织、企事业单位负责人
2-02-01-01,软件工程技术人员,2,专业技术人员
...

如果你没有现成的CSV，脚本会自动进入"第一步：生成职业列表"模式，
用DeepSeek生成全部1636+110个职业的基础信息。
"""

import os
import json
import csv
import time
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("请先安装 openai 库: pip install openai")
    sys.exit(1)

# ========== 配置 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # DeepSeek官方API地址
MODEL = "deepseek-chat"  # 或 deepseek-reasoner

BATCH_SIZE = 20       # 每批处理的职业数量
SLEEP_BETWEEN = 1     # 每批之间的等待秒数（避免限流）
OUTPUT_FILE = "occupations_full.csv"
CHECKPOINT_FILE = "checkpoint.json"  # 断点续传文件

# ========== DeepSeek 客户端 ==========
def get_client():
    if not DEEPSEEK_API_KEY:
        print("\n❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        print("   export DEEPSEEK_API_KEY='sk-xxxxxxxxxxxxxxxx'")
        print("   或在脚本中直接修改 DEEPSEEK_API_KEY 变量")
        sys.exit(1)
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def call_deepseek(client, prompt, system_prompt="你是一个中国劳动市场数据分析专家。", max_retries=3):
    """调用DeepSeek API，带重试"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    return None


# ========== 第一步：生成职业列表（如果没有现成CSV） ==========
def generate_occupation_list(client):
    """用DeepSeek生成全部职业列表"""
    print("\n📋 第一步：生成中国职业分类大典完整职业列表...")
    print("   这需要多次API调用，请耐心等待...\n")
    
    all_occupations = []
    
    # 按大类逐个生成
    categories = {
        1: ("党的机关、国家机关、群众团体和社会组织、企事业单位负责人", 80),
        2: ("专业技术人员", 500),
        3: ("办事人员和有关人员", 150),
        4: ("社会生产服务和生活服务人员", 300),
        5: ("农、林、牧、渔业生产及辅助人员", 100),
        6: ("生产制造及有关人员", 400),
        7: ("军人", 10),
        8: ("不便分类的其他从业人员", 30),
    }
    
    for cat_id, (cat_name, approx_count) in categories.items():
        print(f"  📂 正在生成第{cat_id}大类: {cat_name} (约{approx_count}个职业)...")
        
        # 分批生成，每次请求生成约100个
        batches_needed = max(1, approx_count // 80)
        
        for batch in range(batches_needed):
            offset = batch * 80
            prompt = f"""请列出中国《中华人民共和国职业分类大典(2022年版)》中第{cat_id}大类"{cat_name}"的所有职业（细类）。
同时包含2023-2025年人社部新增的属于此大类的新职业。

请从第{offset+1}个开始，列出最多80个职业。

请严格按以下JSON数组格式输出，不要输出任何其他内容：
[
  {{"code": "{cat_id}-XX-XX-XX", "name": "职业名称"}},
  ...
]

注意：
- code格式为 大类-中类-小类-细类，如 2-02-01-01
- 如果不确定精确的code，可以按顺序编号
- 确保覆盖该大类下的所有职业，不要遗漏
- 包含2023年、2024年、2025年人社部新公布的新职业"""

            result = call_deepseek(client, prompt)
            if result:
                try:
                    # 清理可能的markdown标记
                    cleaned = result.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("\n", 1)[1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned.rsplit("```", 1)[0]
                    cleaned = cleaned.strip()
                    
                    items = json.loads(cleaned)
                    for item in items:
                        all_occupations.append({
                            "code": item.get("code", f"{cat_id}-00-00-{len(all_occupations):02d}"),
                            "name": item.get("name", ""),
                            "major_category": cat_id,
                            "major_category_name": cat_name,
                        })
                    print(f"    ✅ 批次{batch+1}: 获取 {len(items)} 个职业")
                except json.JSONDecodeError as e:
                    print(f"    ❌ JSON解析失败: {e}")
                    print(f"    原始返回: {result[:200]}...")
            
            time.sleep(SLEEP_BETWEEN)
    
    # 去重
    seen = set()
    unique = []
    for occ in all_occupations:
        key = occ["name"]
        if key not in seen:
            seen.add(key)
            unique.append(occ)
    
    # 保存基础列表
    base_file = "occupations_input.csv"
    with open(base_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "name", "major_category", "major_category_name"])
        writer.writeheader()
        writer.writerows(unique)
    
    print(f"\n✅ 职业列表生成完成！共 {len(unique)} 个职业")
    print(f"   已保存到: {base_file}")
    return unique


# ========== 第二步：批量生成AI评分和详细数据 ==========
def generate_scores_batch(client, occupations, start_idx=0):
    """批量生成AI评分、薪资、就业人数、描述"""
    print(f"\n🤖 第二步：为 {len(occupations)} 个职业生成AI冲击评分...")
    print(f"   每批 {BATCH_SIZE} 个，预计需要 {len(occupations)//BATCH_SIZE+1} 次API调用\n")
    
    # 加载断点
    results = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"   📌 从断点恢复，已完成 {len(results)} 个职业\n")
    
    total = len(occupations)
    
    for i in range(0, total, BATCH_SIZE):
        batch = occupations[i:i+BATCH_SIZE]
        
        # 跳过已完成的
        batch_to_process = [occ for occ in batch if occ["name"] not in results]
        if not batch_to_process:
            continue
        
        names_list = "\n".join([f"{j+1}. {occ['name']}（{occ['major_category_name']}）" for j, occ in enumerate(batch_to_process)])
        
        prompt = f"""请为以下中国职业评估AI冲击程度，并估算相关数据。

职业列表：
{names_list}

请为每个职业返回JSON数组，包含以下字段：
- name: 职业名称
- score: AI冲击指数(0-10, 0=完全不受影响, 10=将被完全替代)
- salary: 2025年估算平均年薪(单位:万元，参考国家统计局2024年行业数据+4.4%增长)
- employment: 估算全国从业人数(单位:万人)
- description: 职业描述(60字以内，描述主要工作内容)

评分标准：
- 0-1: 纯体力/物理操作，如屋顶工、清洁工
- 2-3: 体力为主+少量信息处理，如电工、护士助理
- 4-5: 人际互动+判断力，如护士、零售、医生
- 6-7: 知识工作+决策，如教师、经理、工程师
- 8-9: 高度数字化工作，如程序员、数据分析、编辑
- 10: 几乎完全可自动化，如医学转录员

请严格只输出JSON数组，格式如下：
[
  {{"name":"XXX","score":7.5,"salary":25,"employment":350,"description":"从事XXX工作"}},
  ...
]"""

        result = call_deepseek(client, prompt)
        if result:
            try:
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned.rsplit("```", 1)[0]
                
                items = json.loads(cleaned.strip())
                for item in items:
                    results[item["name"]] = {
                        "score": item.get("score", 5),
                        "salary": item.get("salary", 8),
                        "employment": item.get("employment", 50),
                        "description": item.get("description", ""),
                    }
                
                done = len(results)
                pct = done / total * 100
                print(f"  ✅ [{done}/{total}] ({pct:.0f}%) 已处理批次 {i//BATCH_SIZE+1}")
                
                # 保存断点
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                    
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON解析失败: {e}")
        else:
            print(f"  ❌ API调用失败，跳过此批次")
        
        time.sleep(SLEEP_BETWEEN)
    
    return results


# ========== 第三步：合并输出最终CSV ==========
def merge_and_output(occupations, scores):
    """合并职业列表和评分数据，输出最终CSV"""
    print(f"\n📊 第三步：合并数据并输出...")
    
    rows = []
    for occ in occupations:
        name = occ["name"]
        data = scores.get(name, {})
        rows.append({
            "code": occ["code"],
            "name": name,
            "major_category": occ["major_category"],
            "major_category_name": occ["major_category_name"],
            "description": data.get("description", ""),
            "score": data.get("score", 5.0),
            "salary": data.get("salary", 8),
            "employment": data.get("employment", 50),
        })
    
    # 保存CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "code", "name", "major_category", "major_category_name",
            "description", "score", "salary", "employment"
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    # 同时保存JSON（方便直接导入前端）
    json_file = OUTPUT_FILE.replace(".csv", ".json")
    json_data = []
    for row in rows:
        json_data.append({
            "c": row["code"],
            "n": row["name"],
            "cat": int(row["major_category"]),
            "cn": row["major_category_name"],
            "d": row["description"],
            "s": float(row["score"]),
            "e": int(row["employment"]),
            "sal": int(row["salary"]),
        })
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, separators=(",", ":"))
    
    # 统计
    scores_list = [r["score"] for r in rows]
    avg_score = sum(scores_list) / len(scores_list) if scores_list else 0
    total_emp = sum(r["employment"] for r in rows)
    
    print(f"\n{'='*50}")
    print(f"✅ 数据生成完成！")
    print(f"   职业总数: {len(rows)}")
    print(f"   平均AI冲击指数: {avg_score:.1f}/10")
    print(f"   覆盖就业人数: {total_emp}万人")
    print(f"   输出文件: {OUTPUT_FILE} (CSV)")
    print(f"   输出文件: {json_file} (JSON)")
    print(f"{'='*50}")
    print(f"\n💡 下一步: 将 {OUTPUT_FILE} 上传到AI饭碗系统即可！")
    
    return rows


# ========== 主程序 ==========
def main():
    print("=" * 50)
    print("🦐 AI饭碗系统 - 职业数据生成器")
    print("   使用 DeepSeek API 生成完整职业数据")
    print("=" * 50)
    
    client = get_client()
    
    # 检查是否有现成的输入文件
    input_file = "occupations_input.csv"
    
    if os.path.exists(input_file):
        print(f"\n📂 发现输入文件: {input_file}")
        occupations = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                occupations.append(row)
        print(f"   共 {len(occupations)} 个职业")
    else:
        print(f"\n📂 未发现输入文件 {input_file}")
        print("   将使用 DeepSeek 自动生成职业列表...\n")
        
        choice = input("是否继续？(y/n): ").strip().lower()
        if choice != "y":
            print("已取消")
            return
        
        occupations = generate_occupation_list(client)
    
    # 生成AI评分
    scores = generate_scores_batch(client, occupations)
    
    # 合并输出
    merge_and_output(occupations, scores)


if __name__ == "__main__":
    main()
