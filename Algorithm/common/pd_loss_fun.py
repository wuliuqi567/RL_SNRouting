import torch
import torch.nn as nn

def kl_distillation_loss(student_outputs, teacher_outputs, temperature):
    """
    计算KL散度损失，用于策略蒸馏。

    loss_kl = \sum_{i = 1}^N softmax(\frac{Q^T}{t} \ln \frac{softmax(\frac{Q^T}{t})} {softmax(\frac{Q^S}{t})})

    参数:
    student_outputs: 学生网络的输出 (logits)
    teacher_outputs: 教师网络的输出 (logits)
    temperature: 温度参数，控制软化程度
    
    返回:
    KL散度损失值
    """
    # 应用温度缩放
    student_logits = student_outputs / 1
    teacher_logits = teacher_outputs / temperature
    
    # 计算softmax概率分布
    student_probs = nn.functional.softmax(student_logits, dim=1)  # softmax(Q^S/t)
    teacher_probs = nn.functional.softmax(teacher_logits, dim=1)  # softmax(Q^T/t)
    
    # 根据公式计算KL散度: ∑ softmax(Q^T/t) * ln(softmax(Q^T/t) / softmax(Q^S/t))
    # 等价于: ∑ teacher_probs * ln(teacher_probs / student_probs)
    # 为了数值稳定性，使用 log(teacher_probs) - log(student_probs)
    log_teacher_probs = torch.log(teacher_probs + 1e-8)  # 加小值避免log(0)
    log_student_probs = torch.log(student_probs + 1e-8)
    
    # KL散度计算: teacher_probs * (log_teacher_probs - log_student_probs)
    kl_loss = torch.sum(teacher_probs * (log_teacher_probs - log_student_probs), dim=1)
    kl_loss = torch.mean(kl_loss) * (temperature ** 2)  # 温度平方缩放，返回标量
    
    return kl_loss

def negative_log_likelihood_loss(student_outputs, teacher_outputs):
    """
    计算负对数似然损失，用于策略蒸馏。

    loss_nll = - \sum_{i = 1}^N argmax(Q^T) \ln softmax(Q^S)

    参数:
    student_outputs: 学生网络的输出 (logits)
    teacher_outputs: 教师网络的输出 (logits)

    返回:
    负对数似然损失值
    """
    # 计算教师网络的动作概率分布
    teacher_probs = nn.functional.softmax(teacher_outputs, dim=1)
    # 获取教师网络选择的动作索引
    _, teacher_actions = torch.max(teacher_probs, dim=1)

    # 计算学生网络的动作概率分布
    student_log_probs = nn.functional.log_softmax(student_outputs, dim=1)

    # 计算负对数似然损失
    nll_loss = nn.NLLLoss()  # 默认 reduction='mean'，已经返回平均值
    loss = nll_loss(student_log_probs, teacher_actions)  # 返回标量

    return loss


def kl_distillation_loss_v2(student_outputs, teacher_outputs, temperature):
    """
    计算双重KL散度损失，用于策略蒸馏。

    loss_kl = \sum_{i = 1}^N softmax(\frac{Q^T}{t} \ln \frac{softmax(\frac{Q^T}{t})} {softmax(\frac{Q^S}{t})})

    参数:
    student_outputs: 学生网络的输出 (logits)
    teacher_outputs: 教师网络的输出 (logits)
    temperature: 温度参数，控制软化程度

    返回:
    KL散度损失值
    """
    kl_loss = kl_distillation_loss(student_outputs, teacher_outputs, temperature)
    nll_loss = negative_log_likelihood_loss(student_outputs, teacher_outputs)
    return nll_loss*0.5 + kl_loss*0.5