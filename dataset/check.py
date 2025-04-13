import json

def check_file(file_path):
    # 存储字段数据类型统计
    type_stats = {}

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            js = json.loads(line)
            for key, value in js.items():
                # 获取字段的数据类型
                dtype = type(value).__name__

                # 统计该字段的数据类型
                if key not in type_stats:
                    type_stats[key] = {}
                if dtype not in type_stats[key]:
                    type_stats[key][dtype] = 0
                type_stats[key][dtype] += 1

    # 输出每个字段的数据类型统计
    print(f"检查文件：{file_path}")
    for key, stats in type_stats.items():
        print(f"\n字段: {key}")
        for dtype, count in stats.items():
            print(f"    {dtype}: {count} 条数据")

# 检查文件
check_file('train.jsonl')


