import pandas as pd
from mlxtend.frequent_patterns import apriori


def apriori_mlxtend(combined_list, min_support=2):
    """
    使用 mlxtend.frequent_patterns.apriori 实现的高性能频繁项集挖掘。

    参数:
        combined_list: list[list] — 事务数据（每个事务是一个部件集合）
        min_support: int 或 float — 若 >=1 表示最小出现次数；若 <1 表示比例支持度

    返回:
        frequent_combined_list: list[list[frozenset]] — 按项集长度划分的频繁项集
        support_data: dict[frozenset -> float] — 每个项集的支持度
    """
    # 将 combined_list 转为 one-hot 编码的 DataFrame
    all_parts = sorted(set().union(*combined_list))
    df = pd.DataFrame(0, index=range(len(combined_list)), columns=all_parts)
    for i, items in enumerate(combined_list):
        df.loc[i, items] = 1

    # 计算最小支持度（若传入整数则换算为比例）
    if min_support >= 1:
        min_support = min_support / len(combined_list)

    # 调用 mlxtend 的 Apriori
    result = apriori(df, min_support=min_support, use_colnames=True)

    # 转换为你的格式
    # 支持度字典
    support_data = {
        frozenset(row['itemsets']): row['support']
        for _, row in result.iterrows()
    }

    # 按项集长度划分
    frequent_combined_list = []
    grouped = result.groupby(result['itemsets'].apply(len))
    for k in sorted(grouped.groups.keys()):
        group_k = [
            frozenset(items)
            for items in grouped.get_group(k)['itemsets']
        ]
        frequent_combined_list.append(group_k)

    return frequent_combined_list, support_data
