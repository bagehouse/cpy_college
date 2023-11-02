import pandas as pd

# 加载CSV表格数据
df = pd.read_csv('data/no/input.CSV',encoding='gbk')

# 存储处理后的数据
output_data = []
row_counter = 0

# 处理每一行数据
while row_counter < len(df):
    # 将当前行的数据转换为列表
    row_data = df.iloc[row_counter, :].tolist()

 

    # 更新计数器
    row_counter += 2

    if row_counter >= len(df):
        break

    # 保留下一行的数据
    row_data = df.iloc[row_counter, :].tolist()
    output_data.append(row_data)

    # 更新计数器
    row_counter += 1


# 将存储处理后的数据的列表转换为DataFrame格式，并保存到新的CSV表格文件中
output_df = pd.DataFrame(output_data)
output_df.to_csv('data/no/output.CSV', index=False, header=False)
