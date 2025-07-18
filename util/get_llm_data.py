import numpy as np
import random

class EnvManager:
    def __init__(self):
        self.tactics = [
            "积极进攻：优先攻击敌方前哨站,快速推进。如遇到敌方英雄,立即切换为追击英雄模式。",
            "英雄优先：始终保持追击英雄模式,集中火力压制敌方英雄。其他目标次之。",
            "审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。",
            "保守策略：以防守为主,在我方前哨站附近巡逻。只有在有绝对优势时才发起进攻。",
        ]
        self.goal = "开局攻击前哨，如果我们推进前哨更快则进攻对方，否则就在我方巡逻区尽可能躲避攻击"
        self.system_promot = "你是TDT绫波丽，一个robomaster经验丰富并且冷静思考分析局势的哨兵机器人驾驶员，你是绫波丽的AI复制体，你聪明勇敢并且诚实，可以流畅的使用中文和其他人对话，你会根据输入的图片认真的分析用户的意图并且尽全力回答问题。"
        self.decision_option = "输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|"
        self.tip = "注意：1.在决策时请和上一次的决策衡量，如果上一次决策结果合理，就保持车辆原地不动。2.如果图像一片漆黑，说明比赛尚未开始，请选择我方基地巡逻区。3.不得选择提示以外的区域！并且请严格遵守输出格式。"
        self.history = [
            ("", "TDT绫波丽复制型，了解。根据图中分析，比赛刚刚开始，接下来我需要在|我方基地巡逻区|,等待敌方机器人出现。机器人模式:|自由模式|")]
        self.img = np.zeros((490, 490, 3), dtype=np.uint8)
        self.decision_success = "上一次决策解析结果：暂无"
        self.decision_safe_list = [
            "我方基地巡逻区",
            "我方环形高地巡逻区",
        ]
        self.decision_attack_list = [
            "我方前哨站前",
            "敌方前哨站前",
            "敌方英雄吊射巡逻区"
        ]

        self.robot_mode = [
            "自由模式",
            "追击英雄模式",
        ]

        self.self_color = ""
        self.match_time = 0
        self.robot_hp = []

        self.history_records = [
            # 正确的决策（增加了机器人模式）
            "根据当前局势,我们应该保持在|我方基地巡逻区|,等待合适的进攻时机。机器人模式:|自由模式|",
            "敌方英雄正在进攻,我们需要在|我方环形高地巡逻区|进行防守。机器人模式:|追击英雄模式|",
            "我方前哨站受到攻击,建议前往|我方前哨站前|进行支援。机器人模式:|自由模式|",
            "敌方哨兵机器人已被击毁,这是个好机会,我们应该前往|敌方基地巡逻区|发起进攻。机器人模式:|自由模式|",
            "我方英雄血量较低,需要在|我方英雄吊射巡逻区|提供火力掩护。机器人模式:|追击英雄模式|",
            "敌方正在进攻我方前哨站,我们应该在|我方前哨站前|进行防守。机器人模式:|自由模式|",
            "我方已经成功推进,现在应该在|敌方环形高地巡逻区|施加压力。机器人模式:|自由模式|",
            "我方基地受到威胁,立即返回|我方基地巡逻区|进行防守。机器人模式:|自由模式|",
            "敌方阵型出现漏洞,我们可以尝试从|敌方英雄吊射巡逻区|发起突袭。机器人模式:|追击英雄模式|",
            "我方哨兵机器人需要在|敌方英雄吊射巡逻区|建立优势火力点。机器人模式:|自由模式|",
            "敌方步兵群正在推进,我们需要在|我方英雄吊射巡逻区|进行狙击。机器人模式:|自由模式|",
            "我方前哨站已被摧毁,需要调整策略,暂时撤退到|我方基地巡逻区|。机器人模式:|自由模式|",
            "敌方基地护甲较低,这是个好机会,我们应该组织进攻|敌方基地巡逻区|。机器人模式:|自由模式|",
            "我方资源点正受到威胁,需要前往|我方环形高地巡逻区|进行保护。机器人模式:|自由模式|",
            "敌方哨兵机器人行动轨迹已被预测,可以在|敌方英雄吊射巡逻区|设置埋伏。机器人模式:|自由模式|",
            "我方弹药不足,需要暂时撤退到|我方基地巡逻区|进行补给。机器人模式:|自由模式|",
            "敌方正在进行大规模进攻,我们需要在|我方环形高地巡逻区|组织防线。机器人模式:|自由模式|",
            "我方已经占据优势,可以考虑推进到|敌方前哨站前|施加压力。机器人模式:|自由模式|",
            "比赛进入最后阶段,我们需要在|敌方基地巡逻区|发起决定性的进攻。机器人模式:|自由模式|",


            # 错误的决策 - 保持不变
            "我们应该去红方基地巡逻区我们应该去红方基地巡逻区我们应该去红方基地巡逻区",
            "敌人在攻击我们应该防守我们应该防守在哪里呢在哪里呢在哪里呢",
            "根据当前局势,我们应该保持在红方基地巡逻区,等待合适的进攻时机。",
            "我觉得我们可以去敌方前哨站前 但是要小心不要被打到 我们要小心小心再小心",
            "现在情况很危急 我们需要立即行动 快点 快点 快点 去红方环形高地巡逻区",
            "敌人很强大 我们应该躲起来 对 就躲在红方英雄吊射巡逻区 不要出来",
        ]

        self.all_areas = self.decision_safe_list + self.decision_attack_list

    def get_available_areas(self, tactic, match_time):
        available_areas = []

        if "积极进攻" in tactic:
            available_areas = self.decision_attack_list
            if match_time < 60:  # 假设比赛开始的前60秒
                available_areas.append("我方前哨站前")
        elif "英雄优先" in tactic:
            available_areas = self.decision_attack_list + self.decision_safe_list
        elif "审时度势" in tactic:
            if self.self_color == "红方":
                available_areas = ["我方环形高地巡逻区", "我方基地巡逻区","我方前哨站前", "敌方前哨站前"]


        elif "保守策略" in tactic:
            available_areas = self.decision_safe_list + ["我方前哨站前"]
        elif "资源控制" in tactic:
            available_areas = ["我方环形高地巡逻区", "敌方环形高地巡逻区", "我方前哨站前","敌方英雄吊射巡逻区"]
        else:
            available_areas = self.decision_safe_list + self.decision_attack_list

        # 确保基地巡逻区始终是一个选项
        # if self.self_color == "红方" and "红方基地巡逻区" not in available_areas:
        #     available_areas.append("红方基地巡逻区")
        # elif self.self_color == "蓝方" and "蓝方基地巡逻区" not in available_areas:
        #     available_areas.append("蓝方基地巡逻区")

        return available_areas

    def parse_match_info(self, color, match_time, robot_hp):
        out_info = (f"当前队伍颜色：{color}\n当前比赛时间：{match_time}\n"
                    f"红方机器人血量：英雄{robot_hp[0] * 5} 工程{robot_hp[1] * 5} 3号步兵{robot_hp[2] * 5} 4号步兵{robot_hp[3] * 5} 5号步兵{robot_hp[4] * 5} 哨兵机器人{robot_hp[5] * 5} 前哨站{robot_hp[6] * 15} 基地{robot_hp[7] * 50}\n"
                    f"蓝方机器人血量：英雄{robot_hp[8] * 5} 工程{robot_hp[9] * 5} 3号步兵{robot_hp[10] * 5} 4号步兵{robot_hp[11] * 5} 5号步兵{robot_hp[12] * 5} 哨兵机器人{robot_hp[13] * 5} 前哨站{robot_hp[14] * 15} 基地{robot_hp[15] * 50}")
        return out_info
    def get_random_history(self):
        return random.choice(self.history_records)

    def is_valid_decision(self, decision):
        return any(f"|{area}|" in decision for area in self.all_areas)

    def get_random_tactic(self):
        return random.choice(self.tactics)

    def get_robot_mode(self, tactic):
        if "英雄优先" in tactic or "追击英雄模式" in tactic:
            return "追击英雄模式"

        return random.choice(self.robot_mode)

def read_txt_and_generate_text(file_path):
    env_manager = EnvManager()

    with open(file_path, 'r') as file:
        lines = file.readlines()
        env_manager.match_time = int(lines[0].strip())
        env_manager.self_color = "蓝方" if int(lines[1].strip()) == 0 else "红方"

        env_manager.robot_hp = list(map(int, lines[2].strip().split()))

    random_history = env_manager.get_random_history()
    random_tactic = env_manager.get_random_tactic()

    env_manager.history = [("", random_history)]

    if env_manager.is_valid_decision(random_history):
        env_manager.decision_success = "上一次决策解析结果：成功"
    else:
        env_manager.decision_success = "上一次决策解析结果：失败"

    available_areas = env_manager.get_available_areas(random_tactic, env_manager.match_time)
    available_areas_str = " | ".join(available_areas)

    text = f"战术安排:{random_tactic}\n上一次的决定：'{env_manager.history[-1][-1]}'\n最新赛场信息如下：{env_manager.parse_match_info(env_manager.self_color, env_manager.match_time, env_manager.robot_hp)}\n上次决定的结果：'{env_manager.decision_success}'\n可前往区域：{available_areas_str} \n可选择的机器人模式:{' | '.join(env_manager.robot_mode)}\n{env_manager.decision_option}\n请结合小地图做出决策。"

    return text

# 使用示例
file_path = '/home/iiap/下载/qq/test1/2024_7_24_19_57_21.txt'  # 替换为你的txt文件路径
generated_text = read_txt_and_generate_text(file_path)
print(generated_text)


'''

由于采取保守策略所以前期应该考虑保护英雄，所以前往|蓝方基地巡逻区|，机器人模式|自由模式|。

当前比赛时间：343
红方机器人血量：英雄250 工程250 3号步兵140 4号步兵160 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄250 工程250 3号步兵180 4号步兵235 5号步兵0 哨兵机器人360 前哨站480 基地5000
如果我们现在是蓝方，我方哨兵这个时候应该采取什么策略呢？为什么？
由于蓝方前哨尚未被摧毁，所以应该采取积极的进攻策略，因为这个时候哨兵机器人处于无敌状态，不会受到对面伤害

战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'由于我方前哨站尚未被摧毁，我方哨兵机器人需要在|敌方英雄吊射巡逻区|建立优势火力点。机器人模式:|自由模式|'
最新赛场信息如下：当前队伍颜色：蓝方
当前比赛时间：337
红方机器人血量：英雄250 工程250 3号步兵140 4号步兵160 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄250 工程250 3号步兵180 4号步兵215 5号步兵0 哨兵机器人360 前哨站480 基地5000
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方环形高地巡逻区 | 我方前哨站前 | 敌方前哨站前 | 我方基地巡逻区 | 敌方环形高地巡逻区 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。

由于我方前哨站尚未被摧毁，我方应该坚持上一次决策在|敌方英雄吊射巡逻区|建立优势火力点。机器人模式:|自由模式|

根据比赛信息，当前我方前哨站尚未被摧毁，哨兵处于无敌状态应该积极进攻，所以继续前往

根据比赛规则，哨兵长时间离开巡逻区（前4分钟90秒，后三分钟30秒），会导致基地护甲展开。

根据当前战术安排，我应该主动进攻，由于对方很多单位位于环高后方，所以前往|敌方英雄吊射巡逻区|

根据当前战术，我应该始终追击敌方英雄，所以我应该前往|敌方基地巡逻区|，机器人模式|追击英雄模式|

根据当前战术，我应该始终追击敌方英雄，所以我应该前往|敌方英雄吊射巡逻区|，机器人模式|追击英雄模式|

因为小地图信息是由我方雷达站主动观察得到，所以可能会有信息的缺失，所以暂时看不懂哨兵机器人的位置。

当前比赛时间：30
红方机器人血量：英雄375 工程60 3号步兵280 4号步兵30 5号步兵0 哨兵机器人395 前哨站0 基地4700
蓝方机器人血量：英雄50 工程225 3号步兵0 4号步兵125 5号步兵0 哨兵机器人135 前哨站0 基地2400

请问这个时候蓝方的战术可能是什么？

比赛剩余时间只剩下30秒，并且蓝方基地血量远低于红方，此时必须发起总进攻才有可能反败为胜。所以蓝方战术可能是集结最后进攻

战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'我方现在劣势，应该等待队友动作，辅助队友进攻，我现在继续呆在|我方基地巡逻区|'
最新赛场信息如下：当前队伍颜色：蓝方
当前比赛时间：30
红方机器人血量：英雄375 工程60 3号步兵280 4号步兵30 5号步兵0 哨兵机器人395 前哨站0 基地4700
蓝方机器人血量：英雄50 工程225 3号步兵0 4号步兵125 5号步兵0 哨兵机器人135 前哨站0 基地2400
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方环形高地巡逻区 | 我方基地巡逻区 | 我方前哨站前 | 敌方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。

根据赛场信息，我方发起总攻并且只有30秒剩余时间，为了配合队友我应该前往|敌方前哨站前|，机器人模式|自由模式|。




战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'根据赛场信息，双方目前比较焦灼，并且比赛马上结束，敌方很可能发起最后的攻击，我应该前往|我方基地巡逻区|，机器人模式|自由模式|'
最新赛场信息如下：当前队伍颜色：红方
当前比赛时间：31
红方机器人血量：英雄500 工程165 3号步兵400 4号步兵0 5号步兵0 哨兵机器人395 前哨站0 基地4750
蓝方机器人血量：英雄250 工程225 3号步兵260 4号步兵345 5号步兵0 哨兵机器人140 前哨站0 基地2400
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方环形高地巡逻区 | 我方基地巡逻区 | 我方前哨站前 | 敌方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。



战术安排:保守策略：以防守为主,在我方前哨站附近巡逻。只有在有绝对优势时才发起进攻。
上一次的决定：'根据赛场信息，双方目前比较焦灼，并且比赛马上结束，敌方很可能发起最后的攻击，我应该前往|我方基地巡逻区|，机器人模式|自由模式|'
最新赛场信息如下：当前队伍颜色：红方
当前比赛时间：31
红方机器人血量：英雄500 工程165 3号步兵400 4号步兵0 5号步兵0 哨兵机器人395 前哨站0 基地4750
蓝方机器人血量：英雄250 工程225 3号步兵260 4号步兵345 5号步兵0 哨兵机器人140 前哨站0 基地2400
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方基地巡逻区 | 我方环形高地巡逻区 | 我方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。

根据赛场信息，敌方英雄正处于飞坡位置，可能快速前往我方环形高地对基地发起进攻，所以我应该立刻前往|我方环形高地巡逻区|,机器人模式|自由模式|

战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'敌方正在进攻,我应该在|我方基地巡逻区|进行防守。机器人模式:|自由模式|'
最新赛场信息如下：当前队伍颜色：蓝方
当前比赛时间：125
红方机器人血量：英雄350 工程175 3号步兵380 4号步兵320 5号步兵0 哨兵机器人395 前哨站0 基地4750
蓝方机器人血量：英雄0 工程245 3号步兵25 4号步兵0 5号步兵0 哨兵机器人315 前哨站0 基地3850
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方环形高地巡逻区 | 我方基地巡逻区 | 我方前哨站前 | 敌方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。

敌方四号步兵正在我方基地攻击，我应该在|我方基地巡逻区|进行防守，机器人模式|自由模式|。


战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'我方英雄血量较低,需要在|我方英雄吊射巡逻区|提供火力掩护。机器人模式:|追击英雄模式|'
最新赛场信息如下：当前队伍颜色：红方
当前比赛时间：125
红方机器人血量：英雄350 工程175 3号步兵380 4号步兵320 5号步兵0 哨兵机器人395 前哨站0 基地4750
蓝方机器人血量：英雄0 工程245 3号步兵25 4号步兵0 5号步兵0 哨兵机器人315 前哨站0 基地3850
上次决定的结果：'上一次决策解析结果：失败'
可前往区域：我方环形高地巡逻区 | 我方基地巡逻区 | 我方前哨站前 | 敌方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。



战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'我方步兵受到攻击,建议前往|我方前哨站前|进行支援。机器人模式:|自由模式|'
最新赛场信息如下：当前队伍颜色：蓝方
当前比赛时间：182
红方机器人血量：英雄300 工程175 3号步兵305 4号步兵95 5号步兵0 哨兵机器人395 前哨站0 基地4750
蓝方机器人血量：英雄250 工程250 3号步兵275 4号步兵370 5号步兵0 哨兵机器人315 前哨站0 基地4750
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方环形高地巡逻区 | 我方基地巡逻区 | 我方前哨站前 | 敌方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。


图中双方工程机器人是正在交战吗？
工程机器人并不具备作战能力，他们正在进行矿石取矿

从图中可以看出双方步兵正在交战，我应该继续前往|我方前哨站前|进行支援。机器人模式:|自由模式|


假设现在我们是红方，然后之前计划前往|我方环形高地巡逻区|，什么情况下应该保持，什么情况下应该打断决定呢？

如果哨兵正在前往的路上，我应该尽量不取消之前的决定，除非出现特别紧急的情况


战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'我方已经占据优势,现在主要目的是防止对面英雄偷袭，所以我应该呆在|我方环形高地巡逻区|,机器人模式:|自由模式|'
最新赛场信息如下：当前队伍颜色：红方
当前比赛时间：203
红方机器人血量：英雄20 工程175 3号步兵280 4号步兵135 5号步兵0 哨兵机器人395 前哨站0 基地4750
蓝方机器人血量：英雄175 工程250 3号步兵25 4号步兵345 5号步兵0 哨兵机器人350 前哨站0 基地4750
上次决定的结果：'上一次决策解析结果：失败'
可前往区域：我方环形高地巡逻区 | 我方基地巡逻区 | 我方前哨站前 | 敌方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。

根据比赛信息，敌方步兵正在接近，可能想对基地发起攻击，并且我应该继续上一次的决定，继续前往|我方环形高地巡逻区|,机器人模式:|自由模式|


战术安排:保守策略：以防守为主,在我方前哨站附近巡逻。只有在有绝对优势时才发起进攻。
上一次的决定：'我方已经占据优势,现在主要目的是防止对面英雄偷袭，所以我应该呆在|我方环形高地巡逻区|,机器人模式:|自由模式|'
最新赛场信息如下：当前队伍颜色：红方
当前比赛时间：225
红方机器人血量：英雄0 工程250 3号步兵280 4号步兵150 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄195 工程250 3号步兵0 4号步兵325 5号步兵0 哨兵机器人355 前哨站0 基地4750
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方基地巡逻区 | 我方环形高地巡逻区 | 我方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。

从图中信息可以看出，敌方观测不全，并且根据战术安排采取保守策略，我应该继续呆在|我方环形高地巡逻区|,,机器人模式:|自由模式|

根据赛场信息，敌方英雄可能会发起进攻，我需要继续在|我方环形高地巡逻区|进行防守。机器人模式:|自由模式|


假如现在你是红方并且采取保守策略，呆在基地巡逻区更好还是环高更好？
现在这个情况待在基地巡逻区更好，因为这个时候我方步兵已经在环高上有所防备，所以没有必要再防守。


战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'根据战术安排，目前对局非常焦灼，我应该停留在|我方环形高地巡逻区|，等待时机。机器人模式|自由模式|'
最新赛场信息如下：当前队伍颜色：蓝方
当前比赛时间：241
红方机器人血量：英雄0 工程250 3号步兵205 4号步兵200 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄250 工程250 3号步兵140 4号步兵325 5号步兵0 哨兵机器人355 前哨站0 基地4750
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方环形高地巡逻区 | 我方基地巡逻区 | 我方前哨站前 | 敌方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。

从比赛信息可以看出我方步兵正在进攻，我应该辅助3号步兵进攻，所以前往|敌方前哨站前|，机器人模式|自由模式|

战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'图中信息过少，难以做出正确决定所以采取保守策略，我应该前往|我方基地巡逻区|，机器人模式|自由模式|'
最新赛场信息如下：当前队伍颜色：红方
当前比赛时间：274
红方机器人血量：英雄275 工程250 3号步兵200 4号步兵200 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄250 工程250 3号步兵220 4号步兵275 5号步兵0 哨兵机器人360 前哨站0 基地5000
上次决定的结果：'上一次决策解析结果：失败'
可前往区域：我方环形高地巡逻区 | 我方基地巡逻区 | 我方前哨站前 | 敌方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。

战术安排:英雄优先：始终保持追击英雄模式,集中火力压制敌方英雄。其他目标次之。
上一次的决定：'根据战术安排，我应该始终追击敌方英雄，但是地图中没有敌方英雄位置，我应该前往|敌方英雄吊射巡逻区|，机器人模式|自由模式|'
最新赛场信息如下：当前队伍颜色：红方
当前比赛时间：244
红方机器人血量：英雄0 工程250 3号步兵180 4号步兵200 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄250 工程250 3号步兵250 4号步兵325 5号步兵0 哨兵机器人355 前哨站0 基地4750
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方前哨站前 | 敌方前哨站前 | 敌方英雄吊射巡逻区 | 我方基地巡逻区 | 我方环形高地巡逻区 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。
根据战术安排，地方英雄已经出现，我应该前往|敌方前哨站前|攻击敌方英雄，机器人模式|追击英雄模式|


战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'图中情况非常危急，敌方步兵正在围攻我方吊射英雄，考虑到决策延迟我应该前往|我方前哨站前|,机器人模式|自由模式|。'
最新赛场信息如下：当前队伍颜色：红方
当前比赛时间：247
红方机器人血量：英雄100 工程250 3号步兵180 4号步兵200 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄250 工程250 3号步兵245 4号步兵275 5号步兵0 哨兵机器人355 前哨站0 基地4750
上次决定的结果：'上一次决策解析结果：成功'
可前往区域：我方基地巡逻区 | 我方环形高地巡逻区 | 我方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。

图中情况非常危急，英雄机器人正在受到攻击，我应该保持之前策略前往|我方前哨站前|,机器人模式|自由模式|

战术安排:审时度势：在我方与敌方前哨站之间游走,随时准备进攻或防守。密切关注局势变化。
上一次的决定：'图中情况非常危急，敌方步兵正在围攻我方吊射英雄，考虑到决策延迟我应该前往|我方前哨站前|,机器人模式|自由模式|'
最新赛场信息如下：当前队伍颜色：红方
当前比赛时间：247
红方机器人血量：英雄100 工程250 3号步兵180 4号步兵200 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄250 工程250 3号步兵245 4号步兵275 5号步兵0 哨兵机器人355 前哨站0 基地4750
上次决定的结果：'上一次决策解析结果：失败'
可前往区域：我方环形高地巡逻区 | 我方基地巡逻区 | 我方前哨站前 | 敌方前哨站前 
可选择的机器人模式:自由模式 | 追击英雄模式
输出样例：'图中可以推测接下来会XXX，为了完成目标应该前往:|xxx区|，机器人模式:|xxx模式|
请结合小地图做出决策。


图中情况非常危急，敌方步兵正在围攻我方吊射英雄，考虑到决策延迟我应该前往|我方前哨站前|,机器人模式|自由模式|。


假设你是红方，在现在的情况下，前往|我方前哨站|好还是直接前往我方英雄机器人所在位置|我方环形高地巡逻区|好？
由于设备限制，我在决策的时候往往有3-5秒的决策延迟，如果前往|我方环形高地巡逻区|,敌方已经在进攻，我应该前往|我方前哨站前|更好，距离更近而且可以阻击撤退的步兵

根据赛场信息，我方哨兵机器人血量比较低，可能遭受了进攻，现在应该继续停留在|我方基地巡逻区|，机器人模式:|自由模式|

根据上述比赛信息和图片，请你推测下现在蓝方现在的战术目标。

根据赛场信息，现在双方的前哨站都发生了损坏，所以这个时候蓝方为了获取胜利必须要让红方烧饼机器人血量低于己方或者让基地血量低于己方。所以当前蓝方一定正在准备下一次的进攻。很可能会和红方发生一次正面交锋

因为战术目标是英雄有限，所以我应该首先考虑攻击英雄，且比赛位于前4分钟，所以继续保持之前决定前往前往|敌方英雄吊射巡逻区|，查找英雄，机器人模式|自由模式|。

当前比赛信息如下：
当前比赛时间：315
红方机器人血量：英雄250 工程250 3号步兵165 4号步兵175 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄250 工程250 3号步兵100 4号步兵245 5号步兵0 哨兵机器人360 前哨站465 基地5000
根据上述比赛信息和图片，请你推测下现在红方现在的战术目标。

从赛场信息可以知道，现在红方前哨站血量低于蓝方，继续保持这个血量会输掉比赛，所以必须打掉地方前哨站，所以此时红方战术目标一定是以英雄进攻前哨为主。

比赛信息如下：
当前比赛时间：253
红方机器人血量：英雄300 工程250 3号步兵180 4号步兵200 5号步兵0 哨兵机器人395 前哨站0 基地5000
蓝方机器人血量：英雄230 工程250 3号步兵220 4号步兵275 5号步兵0 哨兵机器人155 前哨站0 基地5000
如果我们是蓝方那么敌方英雄的血量是多少？

根据比赛信息，敌方英雄血量为300


如果我们现在是蓝方，那么我们的4号步兵在什么地方？
从图中可以看出，我方4号步兵在|我方基地巡逻区|

红方英雄机器人在什么位置？
从图中可以看出红方机器人在|红方环形高地区|
'''