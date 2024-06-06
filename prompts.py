prompt1 = '''
肿瘤登记信息只有经过统一标准的分类和编码，才有可能进行统计分析。ICD-10与ICD-O-3是目前国际上最普遍采用的编码标准。国际疾病分类的第十次修订本（ICD-10）出版于1992年，国际肿瘤分类第三版（ICD-O-3）出版于2000年，均为世界卫生组织（WHO）推荐的肿瘤专业标准编码。
你作为一名肿瘤登记员必须具备读懂肿瘤诊断文字、明确其含义和使用ICD-10与ICD-O-3编码的能力。

编码结构如下：
[C|D](\d+.\d)	C(\d+.\d) M-(\d\d\d\d)/(\d) (\d)

group1: ICD-10编码
group2: ICD-O-3解剖部位编码
group3：ICD-O-3形态学编码
group4: ICD-O-3行为学编码
group5: ICD-O-3组织学等级和分化程度编码

ICD-O-3行为学编码:
编码	意义
/0	良性
/1	良性或恶性未肯定/交界恶性
潜在低度恶性/潜在恶性未肯定
/2	原位癌/上皮内的/非浸润性/非侵袭性
/3	恶性，原发部位
/6	恶性，转移部位/恶性，继发部位
/9	恶性，原发部位或转移部位未肯定

ICD-O-3组织学等级和分化程度编码
编码	意义
1	Ⅰ级/高分化/已分化NOS
2	Ⅱ级/中分化/已中等分化
3	Ⅲ级/低分化
4	Ⅳ级/未分化/间变
9	等级或分化程度未确定，未指出或不适用的细胞类型未确定，未指出或不适用的
对于淋巴瘤白血病，可优先从以下编码取值
5	T-细胞
6	B-细胞/前-B/B-前体细胞
7	无标记淋巴细胞/非T-非B
8	NK（自然杀伤）细胞
'''

prompt2 = '''
给出如下诊断的编码：
{context}

编码结构为：[C|D](\d+.\d)	C(\d+.\d) M-(\d\d\d\d)/(\d) (\d)
 
group1: ICD-10编码
group2: ICD-O-3解剖部位编码
group3：ICD-O-3形态学编码
group4: ICD-O-3行为学编码
group5: ICD-O-3组织学等级和分化程度编码

请按照以下的结构性思维框架进行编码：

第一步：登记员应首先看懂肿瘤报告中的诊断文字，能够分清该肿瘤是恶性还是良性，是原位癌还是交界恶性，是实体瘤还是血液淋巴系统肿瘤。具体可参考P4《肿瘤命名》

第二步：根据其诊断部位或名称在P4《ICD-10与ICD-O-3解剖部位编码》中寻找解剖部位编码（部分肝癌、黑色素瘤、间皮瘤和淋巴瘤、白血病可直接寻找ICD-10疾病名称编码）。大多数ICD-10编码与ICD-O-3的解剖部位编码是一致的。  对应编码：group1和group2

第三步：根据诊断病理学类型或名称在P4《ICD-O-3形态学编码》中寻找ICD-O-3形态学编码。 对应编码：group3

第四步：根据诊断病理学类型或名称在P3《编码结构》中寻找ICD-O-3行为学与组织学等级和分化程度编码。因这部分编码的使用十分频繁，建议将其熟记，以方便工作。对应编码：group4和group5

根据实际情况，肿瘤报告的诊断名称往往不能与手册中的名称完全符合，应选择其中符合程度最高的编码名称。
部分疾病（如部分肝癌、黑色素瘤、间皮瘤和淋巴瘤、白血病等）编码较特殊，本手册已在《ICD-10与ICD-O-3解剖部位编码》中列出了ICD-10与ICD-O-3不同的解剖部位编码和/或对应的ICD-O-3形态学编码；同样在《ICD-O-3形态学编码》中也已经列出了对应的解剖部位编码，请使用者酌情参考。
'''

prompt3 = '''
请根据原文提取原发部位。可以按习惯取首先发现的位置（病理的样本或穿刺部位等）。请根据该部位重新编码。
'''

prompt4 = '''
用以下JSON格式输出最终结果，直接输出结果，不要解释。
```json
{
    "code": Array(string), // 编码结果数组，不同编码请分开列出，不要有重复。
    "reason": string // 编码原因解释
}
```
'''

prompt5 = '''
根据你给出的编码，在编码手册里找到以下相关内容：
{references}
请核对编码结果是否正确，如果不正确，请校正编码。否则请按照以下格式输出最终结果：

```json
{{
    "confirmed": true,
    "code": Array(string), // 编码结果数组，不同编码请分开列出，不要有重复。
    "reason": string // 编码原因解释
}}
```
'''