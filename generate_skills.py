"""
AI饭碗系统 - 职业AI技能包生成器
================================
为每个职业生成：技能清单 + 推荐工具 + Prompt模板 + Workflow建议

使用方法:
1. 确保已安装 openai: pip3 install openai
2. 设置环境变量: export DEEPSEEK_API_KEY="sk-你的key"
3. 确保 occupations_full.json 在同目录下
4. 运行: python3 generate_skills.py
5. 输出: skills_output.json（完整技能包数据）

脚本会自动断点续传，中断后重新运行会跳过已完成的职业。
"""

import os
import json
import time
import sys

try:
    from openai import OpenAI
except ImportError:
    print("请先安装 openai: pip3 install openai")
    sys.exit(1)

# ========== 配置 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

BATCH_SIZE = 5          # 每批处理5个职业（技能包内容较多，batch小一点）
SLEEP_BETWEEN = 1.5     # 每批间隔
INPUT_FILE = "occupations_full.json"
OUTPUT_FILE = "skills_output.json"
CHECKPOINT_FILE = "skills_checkpoint.json"

# ========== 客户端 ==========
def get_client():
    if not DEEPSEEK_API_KEY:
        print("\n❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        print("   export DEEPSEEK_API_KEY='sk-你的key'")
        sys.exit(1)
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def call_deepseek(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一个AI应用专家，精通各行业的AI工具和自动化工作流。你的任务是为每个职业设计可直接使用的AI技能包。输出必须是严格的JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    return None


def generate_skills_batch(client, batch):
    """为一批职业生成技能包"""
    
    jobs_desc = ""
    for i, job in enumerate(batch):
        jobs_desc += f"\n{i+1}. 职业：{job['n']}\n   行业：{job['cn']}\n   描述：{job['d']}\n   AI冲击指数：{job['s']}/10\n"
    
    prompt = f"""请为以下{len(batch)}个中国职业，每个生成一个完整的AI技能包。

{jobs_desc}

每个职业的技能包必须包含：

1. skills：3-5个该职业最实用的AI技能（每个技能包含name名称、desc一句话说明用AI做什么、tool推荐的具体AI工具名、difficulty难度easy/medium/hard）

2. prompts：2-3个可以直接复制使用的Prompt模板（每个包含title标题、prompt完整的prompt内容，要具体实用，不少于50字，用户复制到ChatGPT/DeepSeek就能直接出结果）

3. workflows：1-2个自动化工作流建议（每个包含title标题、platform推荐平台如Dify/Coze/n8n/飞书多维表格/钉钉等、steps包含3-5个步骤的字符串数组、desc一句话说明效果）

4. free_resources：2-3个免费学习资源（每个包含title名称、type类型如视频/文章/课程/社区、url具体网址或平台名称）

请严格输出JSON数组，格式如下（不要输出任何其他内容，不要markdown标记）：
[
  {{
    "name": "职业名称",
    "skills": [
      {{"name": "技能名", "desc": "用AI做什么", "tool": "推荐工具", "difficulty": "easy"}}
    ],
    "prompts": [
      {{"title": "Prompt标题", "prompt": "完整的prompt内容，用户可直接复制使用..."}}
    ],
    "workflows": [
      {{"title": "工作流标题", "platform": "Dify", "steps": ["步骤1", "步骤2", "步骤3"], "desc": "实现什么效果"}}
    ],
    "free_resources": [
      {{"title": "资源名", "type": "视频", "url": "B站/YouTube/具体平台"}}
    ]
  }}
]"""

    result = call_deepseek(client, prompt)
    if not result:
        return None
    
    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        
        items = json.loads(cleaned)
        return items
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON解析失败: {e}")
        # Try to salvage partial results
        try:
            # Sometimes the JSON is truncated, try to find valid items
            if cleaned.startswith("["):
                # Add closing bracket and try again
                for end_pos in range(len(cleaned)-1, 0, -1):
                    if cleaned[end_pos] == '}':
                        try:
                            items = json.loads(cleaned[:end_pos+1] + "]")
                            print(f"  🔧 部分恢复成功: {len(items)}个职业")
                            return items
                        except:
                            continue
        except:
            pass
        return None


def main():
    print("=" * 55)
    print("🦐 AI饭碗系统 - 职业AI技能包生成器")
    print("   为每个职业生成Skill + Prompt + Workflow")
    print("=" * 55)
    
    client = get_client()
    
    # Load occupations
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ 未找到 {INPUT_FILE}")
        print("   请先运行 generate_occupations.py 生成职业数据")
        sys.exit(1)
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        occupations = json.load(f)
    
    print(f"\n📂 加载 {len(occupations)} 个职业")
    
    # Load checkpoint
    results = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"📌 从断点恢复，已完成 {len(results)} 个职业")
    
    # Filter out already processed
    todo = [occ for occ in occupations if occ['n'] not in results]
    total = len(occupations)
    done = len(results)
    
    print(f"📋 待处理: {len(todo)} 个职业")
    print(f"   每批 {BATCH_SIZE} 个，预计 {len(todo)//BATCH_SIZE+1} 次API调用")
    print(f"   预计耗时: {(len(todo)//BATCH_SIZE+1)*2.5/60:.0f} 分钟\n")
    
    # Process in batches
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i+BATCH_SIZE]
        
        items = generate_skills_batch(client, batch)
        
        if items:
            for item in items:
                name = item.get("name", "")
                if name:
                    results[name] = {
                        "skills": item.get("skills", []),
                        "prompts": item.get("prompts", []),
                        "workflows": item.get("workflows", []),
                        "free_resources": item.get("free_resources", []),
                    }
            
            done = len(results)
            pct = done / total * 100
            print(f"  ✅ [{done}/{total}] ({pct:.0f}%) 批次 {i//BATCH_SIZE+1}")
            
            # Save checkpoint
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            print(f"  ❌ 批次 {i//BATCH_SIZE+1} 失败，跳过")
        
        time.sleep(SLEEP_BETWEEN)
    
    # Merge and output
    print(f"\n📊 合并输出...")
    
    output = []
    matched = 0
    for occ in occupations:
        name = occ['n']
        skill_data = results.get(name, {})
        
        entry = {
            "code": occ['c'],
            "name": name,
            "category": occ['cat'],
            "category_name": occ['cn'],
            "description": occ['d'],
            "ai_score": occ['s'],
            "employment": occ['e'],
            "salary": occ['sal'],
            "skills": skill_data.get("skills", []),
            "prompts": skill_data.get("prompts", []),
            "workflows": skill_data.get("workflows", []),
            "free_resources": skill_data.get("free_resources", []),
        }
        output.append(entry)
        if skill_data:
            matched += 1
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Stats
    total_skills = sum(len(o["skills"]) for o in output)
    total_prompts = sum(len(o["prompts"]) for o in output)
    total_workflows = sum(len(o["workflows"]) for o in output)
    
    print(f"\n{'='*55}")
    print(f"✅ AI技能包生成完成！")
    print(f"   职业总数: {len(output)}")
    print(f"   已生成技能包: {matched}")
    print(f"   技能总数: {total_skills}")
    print(f"   Prompt模板总数: {total_prompts}")
    print(f"   工作流总数: {total_workflows}")
    print(f"   输出文件: {OUTPUT_FILE}")
    print(f"{'='*55}")
    
    # Print a sample
    sample = next((o for o in output if len(o["skills"]) > 0), None)
    if sample:
        print(f"\n📋 示例 - {sample['name']}:")
        print(f"   AI冲击: {sample['ai_score']}/10")
        print(f"   技能:")
        for s in sample['skills'][:3]:
            print(f"     • {s['name']} ({s['tool']}) - {s['desc']}")
        if sample['prompts']:
            print(f"   Prompt模板:")
            for p in sample['prompts'][:2]:
                print(f"     • {p['title']}")
                print(f"       {p['prompt'][:80]}...")
        if sample['workflows']:
            print(f"   工作流:")
            for w in sample['workflows'][:1]:
                print(f"     • {w['title']} ({w['platform']})")
                for step in w['steps'][:3]:
                    print(f"       → {step}")
    
    print(f"\n💡 下一步: 将 {OUTPUT_FILE} 上传到AI饭碗系统即可！")
    print(f"   文件大小约 {os.path.getsize(OUTPUT_FILE)/1024/1024:.1f}MB")


if __name__ == "__main__":
    main()
