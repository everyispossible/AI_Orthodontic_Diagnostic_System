import numpy as np

# ==========================================
# 1. 核心数学与投影工具库
# ==========================================

def get_vector(start, end):
    return np.array(end) - np.array(start)

def calculate_angle(v1, v2):
    cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def calculate_distance(p1, p2, pixel_spacing=1.0):
    return np.linalg.norm(np.array(p2) - np.array(p1)) * pixel_spacing

def signed_dist_to_line(p, line_a, line_b, N, S, pixel_spacing=1.0):
    ab = np.array(line_b) - np.array(line_a)
    facing_right = N[0] > S[0]
    normal1, normal2 = np.array([-ab[1], ab[0]]), np.array([ab[1], -ab[0]])
    normal = normal1 if (normal1[0] > 0) == facing_right else normal2
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    return float(np.dot(np.array(p) - np.array(line_a), normal)) * pixel_spacing

# ==========================================
# 2. 核心测量指标计算
# ==========================================

def calc_ANB(S, N, A, B):
    vec_NS, vec_NA, vec_NB = get_vector(N, S), get_vector(N, A), get_vector(N, B)
    return calculate_angle(vec_NS, vec_NA) - calculate_angle(vec_NS, vec_NB)

def calc_Wits_FOP(A, B, UPM, LPM, UMT, LMT, pixel_spacing=1.0):
    PM = (np.array(LPM) + np.array(UPM)) / 2.0
    M  = (np.array(LMT) + np.array(UMT)) / 2.0
    vec_FOP = PM - M
    dir_FOP = vec_FOP / (np.linalg.norm(vec_FOP) + 1e-8)
    AO = np.dot(np.array(A) - M, dir_FOP)
    BO = np.dot(np.array(B) - M, dir_FOP)
    return float(AO - BO) * pixel_spacing

def calc_E_line_upper(Pn, Pog_soft, Ls, N, S, pixel_spacing=1.0):
    return signed_dist_to_line(Ls, Pn, Pog_soft, N, S, pixel_spacing)

def calc_McNamara_maxilla(N, A, Po, Or, pixel_spacing=1.0):
    vec_FH = np.array(Or) - np.array(Po)
    FH_dir = vec_FH / (np.linalg.norm(vec_FH) + 1e-8)
    return float(np.dot(np.array(A) - np.array(N), FH_dir)) * pixel_spacing

# ==========================================
# 3. 阈值分类器（依据计算值 → 定性结论）
# ==========================================

def _classify_ANB(value):
    """
    返回: (骨性分类标签, 偏差描述, 严重程度 'normal'|'mild'|'severe')
    """
    if value > 4.0:
        severity = "severe" if value > 7.0 else "mild"
        return "骨性 II 类", f"ANB = {value:.1f}°，上颌前突或下颌后缩（正常参考 0°–4°）", severity
    elif value < 0.0:
        severity = "severe" if value < -3.0 else "mild"
        return "骨性 III 类", f"ANB = {value:.1f}°，下颌前突或上颌后缩（正常参考 0°–4°）", severity
    else:
        return "骨性 I 类", f"ANB = {value:.1f}°，颌骨矢状向关系基本正常（正常参考 0°–4°）", "normal"

def _classify_E_line(ls_val, li_val=None):
    """
    ls_val: 上唇到 E 线距离（正 = 唇前于 E 线）
    li_val: 下唇到 E 线距离（可选，若同时传入则综合判断）
    返回: (侧貌标签, 描述, 拔牙倾向 'support'|'caution'|'neutral')
    """
    # 仅上唇判断（li_val 缺失时）
    val = ls_val
    if val > 0:
        label = "上唇前突"
        desc  = f"上唇 E 线距离 = {val:.1f} mm（> 0），唇位偏前"
        tendency = "support"   # 支持拔牙内收
    elif val < -4.0:
        label = "上唇过度后缩"
        desc  = f"上唇 E 线距离 = {val:.1f} mm（< -4 mm），唇位明显靠后"
        tendency = "caution"   # 严控拔牙
    else:
        label = "上唇位置正常"
        desc  = f"上唇 E 线距离 = {val:.1f} mm（-4 ~ 0 mm），处于正常范围"
        tendency = "neutral"
    return label, desc, tendency

def _classify_Wits(value, gender="F"):
    """
    返回: (骨性倾向标签, 描述, 'class2'|'class3'|'normal')
    """
    threshold_class2 = 0.0 if gender == "M" else -1.0
    threshold_class3 = -2.0 if gender == "M" else -3.0

    if value > threshold_class2:
        return "骨性 II 类倾向", f"Wits = {value:.1f} mm（> {threshold_class2} mm），A 点投影前于 B 点", "class2"
    elif value < threshold_class3:
        return "骨性 III 类倾向", f"Wits = {value:.1f} mm（< {threshold_class3} mm），B 点投影前于 A 点", "class3"
    else:
        return "正常范围", f"Wits = {value:.1f} mm，颌骨矢状向关系基本正常", "normal"

def _classify_McNamara(value):
    """
    McNamara 上颌突距（正常参考：女 0–1 mm，男 1–2 mm，此处用中性±2 mm 为界）
    返回: (突度标签, 描述)
    """
    if value > 2.0:
        return "上颌前突", f"McNamara 上颌突距 = {value:.1f} mm（> 2 mm），上颌相对前突"
    elif value < -2.0:
        return "上颌后缩", f"McNamara 上颌突距 = {value:.1f} mm（< -2 mm），上颌相对后缩"
    else:
        return "上颌位置正常", f"McNamara 上颌突距 = {value:.1f} mm，处于正常参考范围"

# ==========================================
# 4. CVM 象限映射
# ==========================================

_CVM_QUADRANT = {
    "CS1": ("象限一", "等待·蓄势期"),
    "CS2": ("象限二", "进攻·黄金期"),
    "CS3": ("象限二", "进攻·黄金期"),
    "CS4": ("象限三", "妥协·代偿期"),
    "CS5": ("象限四", "终局·定型期"),
    "CS6": ("象限四", "终局·定型期"),
}

def _cvm_desc(cvm_stage):
    q, label = _CVM_QUADRANT.get(cvm_stage, ("—", cvm_stage))
    return f"{cvm_stage}（{q} {label}）"

# ==========================================
# 5. 诊断矩阵与 CVM 时空动态路由引擎
# ==========================================

class DiagnosticEngine:
    def __init__(self):
        self.SDR_TABLE = {
            "ANB":      84.0,
            "E_line":   83.3,
            "U1_NA":    76.0,
            "McNamara": 68.7,
            "Wits":     65.3,
            "FMA":      48.7,
            "IMPA":     48.7,
            "FHR":      48.7,
        }

    def get_tier(self, name):
        sdr = self.SDR_TABLE.get(name, 0)
        if sdr > 80:  return 1
        if sdr >= 60: return 2
        return 3

    # ------------------------------------------------------------------
    # 核心路由：依据计算值 → 定性分类 → 结合 CVM 阶段 → 生成临床建议
    # ------------------------------------------------------------------
    def run_cvm_routing(self, name, value, cvm_stage, all_results, gender="F"):
        tier   = self.get_tier(name)
        cvm_d  = _cvm_desc(cvm_stage)
        action = {"status": "", "clinical_advice": "", "system_action": ""}

        # ── 🔴 第三层：全周期冻结，无需数值判断 ──────────────────────
        if tier == 3:
            action["status"]        = "🔴 冻结 (Locked)"
            action["system_action"] = "AI 停止计算，强制触发弹窗，交由人工接管标定（依赖 Go 点，SDR < 60%）"
            fence = "最高等级风险围栏：" if cvm_stage in ["CS2", "CS3"] else ""
            action["clinical_advice"] = (
                f"{fence}{cvm_d} 阶段 {name} 依赖 Go 点定位，"
                "当前 AI 精度不足以支撑临床决策，请手工标定下颌角并建立垂直向/旋转基线。"
            )
            return action

        # ── 🟢 第一层：ANB ──────────────────────────────────────────
        if tier == 1 and name == "ANB":
            action["status"] = "🟢 自动 (Auto)"
            skeletal_class, val_desc, severity = _classify_ANB(value)

            # 共用的数值摘要
            base = f"[{skeletal_class}] {val_desc}。"

            if cvm_stage == "CS1":
                action["system_action"]   = "输出数值，强制召回医生复核生长发育状态"
                if skeletal_class == "骨性 I 类":
                    action["clinical_advice"] = base + "等待期骨性关系正常，以牙性问题评估为主；建议 6–12 个月 CVM 随访，待生长高峰再评估矫治时机。"
                elif skeletal_class == "骨性 II 类":
                    action["clinical_advice"] = base + f"等待期骨改建潜力充足，{'严重' if severity == 'severe' else '轻中度'}Ⅱ类建议纳入 CVM 动态随访队列，暂缓重型功能矫治器，待进入 CS2/3 再启动。"
                else:
                    action["clinical_advice"] = base + "等待期Ⅲ类患者前牵引窗口期最佳，建议尽早评估是否启动上颌前牵引；禁用加重下颌前伸的矫治器。"

            elif cvm_stage in ["CS2", "CS3"]:
                action["system_action"]   = "输出数值，强制触发医生复核 CVM 分期，防止时机偏移"
                if skeletal_class == "骨性 I 类":
                    action["clinical_advice"] = base + "黄金期骨性关系正常，矫治聚焦牙列排列；若有间隙/拥挤问题，此阶段骨支持良好，可积极推进。"
                elif skeletal_class == "骨性 II 类":
                    action["clinical_advice"] = base + f"黄金期{'严重' if severity == 'severe' else '轻中度'}Ⅱ类，提升功能矫治器/Ⅱ类颌间牵引权重，全力利用骨改建窗口；强制医生确认 CVM 分期无误。"
                else:
                    action["clinical_advice"] = base + "黄金期Ⅲ类，前牵引/面罩最后窗口，错过则丧失骨改建机会；强制医生确认 CVM 分期，防止误判导致过早放弃改建。"

            elif cvm_stage in ["CS4", "CS5"]:
                action["system_action"]   = "输出数值，需医生最终确认"
                if skeletal_class == "骨性 I 类":
                    action["clinical_advice"] = base + f"{cvm_stage} 期骨性关系正常，以固定矫治排齐为主；骨改建潜力已{'明显下降' if cvm_stage == 'CS5' else '降低'}。"
                elif skeletal_class == "骨性 II 类":
                    action["clinical_advice"] = base + f"{cvm_stage} 期{'严重' if severity == 'severe' else ''}Ⅱ类骨改建潜力{'基本耗尽，建议成人正颌外科评估' if cvm_stage == 'CS5' else '下降，优先掩饰性代偿（拔牙/非拔牙）'}。"
                else:
                    action["clinical_advice"] = base + f"{cvm_stage} 期Ⅲ类生长改建机会已失，掩饰性正畸为主；{'严重骨性' if severity == 'severe' else '轻度'}差异建议正颌外科联合会诊。"

            elif cvm_stage == "CS6":
                action["system_action"]   = "✅ 安全放行，自动生成结论，免除冗余医生核对"
                if skeletal_class == "骨性 I 类":
                    action["clinical_advice"] = base + "终局期骨性关系正常，成人固定矫治方案适用，无骨改建需求。"
                elif skeletal_class == "骨性 II 类":
                    action["clinical_advice"] = base + f"终局期{'严重' if severity == 'severe' else ''}Ⅱ类骨骼已定型，{'强烈建议正颌外科评估' if severity == 'severe' else '成人掩饰性矫治或正颌外科二选一'}。"
                else:
                    action["clinical_advice"] = base + f"终局期{'严重' if severity == 'severe' else ''}Ⅲ类骨骼定型，{'建议正颌外科联合矫治' if severity == 'severe' else '评估掩饰性成人正畸可行性'}。"
            return action

        # ── 🟢 第一层：E_line ─────────────────────────────────────
        if tier == 1 and name == "E_line":
            action["status"] = "🟢 自动 (Auto)"
            label, val_desc, tendency = _classify_E_line(value)
            base = f"[{label}] {val_desc}。"

            if cvm_stage == "CS1":
                action["system_action"]   = "输出数值，强制召回医生复核"
                if tendency == "support":
                    action["clinical_advice"] = base + "等待期唇位偏前，因垂直向发育未完成，禁止据此做不可逆拔牙决策；纳入随访，待 CS2/3 再评估。"
                elif tendency == "caution":
                    action["clinical_advice"] = base + "等待期唇位已偏后，严格限制任何内收性治疗，防止软组织进一步内陷。"
                else:
                    action["clinical_advice"] = base + "等待期唇位正常，定期随访监测软组织变化趋势。"

            elif cvm_stage in ["CS2", "CS3"]:
                action["system_action"]   = "输出数值，强制触发医生复核"
                if tendency == "support":
                    action["clinical_advice"] = base + "黄金期唇前突，建议联合生长改建（如上颌扩弓/II 类颌间牵引）改善凸面型，减少后期拔牙量；医生复核 CVM 分期确认时机。"
                elif tendency == "caution":
                    action["clinical_advice"] = base + "黄金期唇位已偏后，生长期应以扩弓/前移为主，严禁拔牙内收。"
                else:
                    action["clinical_advice"] = base + "黄金期唇位正常，此阶段关注骨性指标及牙列排列，软组织暂无干预需求。"

            elif cvm_stage in ["CS4", "CS5"]:
                action["system_action"]   = "输出数值，需医生最终确认"
                if tendency == "support":
                    action["clinical_advice"] = base + f"{cvm_stage} 期唇前突是拔牙内收的重要支持依据，需联合牙周状况、拥挤度及唇部张力综合评估拔牙方案。"
                elif tendency == "caution":
                    action["clinical_advice"] = base + f"{cvm_stage} 期唇位已偏后，严控拔牙；若仍需内收，需配合颏成形或唇部软组织评估，防止面型进一步损害。"
                else:
                    action["clinical_advice"] = base + f"{cvm_stage} 期唇位正常，拔牙与否主要参考牙列拥挤度和骨性指标，软组织暂不构成限制因素。"

            elif cvm_stage == "CS6":
                action["system_action"]   = "✅ 安全放行，自动生成结论"
                if tendency == "support":
                    action["clinical_advice"] = base + "终局期唇前突，支持拔牙内收方案；鼻尖形态和头位姿态会影响 E 线，正颌外科设计时保留人工复核接口。"
                elif tendency == "caution":
                    action["clinical_advice"] = base + "终局期唇位偏后，成人矫治以非拔牙或前移方案为主；正颌外科需考虑颏部外形。"
                else:
                    action["clinical_advice"] = base + "终局期唇位正常，成人矫治方案以牙列为主，软组织条件良好。"
            return action

        # ── 🟠 第二层：Wits ──────────────────────────────────────
        if tier == 2 and name == "Wits":
            action["status"]        = "🟠 校验 (Verify)"
            action["system_action"] = "输出倾向性结论，强制附加『需医生人工核对』标签"

            wits_label, wits_desc, wits_class = _classify_Wits(value, gender)
            base = f"[{wits_label}] {wits_desc}。"

            # 检查与 ANB 是否冲突
            anb_val    = all_results.get("ANB")
            conflict   = False
            anb_class  = None
            if anb_val is not None:
                _, _, anb_severity = _classify_ANB(anb_val)
                is_anb_class2 = anb_val > 4.0
                is_anb_class3 = anb_val < 0.0
                anb_class = "II" if is_anb_class2 else ("III" if is_anb_class3 else "I")
                conflict = (is_anb_class2 and wits_class == "class3") or \
                           (is_anb_class3 and wits_class == "class2")

            if cvm_stage in ["CS2", "CS3"] and conflict:
                action["status"]        = "⚠️ 动态降权 (Downgraded)"
                action["system_action"] = "强制触发人工校准咬合平面投影"
                action["clinical_advice"] = (
                    f"黄金期注意：{base}"
                    f"与高置信度 ANB（骨性 {anb_class} 类，ANB = {anb_val:.1f}°）定性冲突，"
                    "已自动降低 Wits 权重，优先参考 ANB。"
                    "请手工校准咬合平面投影，确认 Wits 测量基准无偏移。"
                )
            else:
                cvm_advice = {
                    "CS1": "等待期建立骨性基线，手工核对咬合平面投影是否水平。",
                    "CS2": "黄金期 Wits 与 ANB 方向一致，骨性差异双重印证，可加强矫形力度推荐。",
                    "CS3": "黄金期 Wits 与 ANB 方向一致，骨性差异双重印证，可加强矫形力度推荐。",
                    "CS4": "代偿期 Wits 用于评估掩饰性移动难度；数值越大，代偿量越大，骨开窗风险越高。",
                    "CS5": "定型期 Wits 作为正颌手术截骨量参考，需医生最终确认所有定量结果。",
                    "CS6": "终局期 Wits 作为正颌手术截骨量参考，需医生最终确认所有定量结果。",
                }.get(cvm_stage, "")
                action["clinical_advice"] = base + cvm_advice
            return action

        # ── 🟠 第二层：McNamara ──────────────────────────────────
        if tier == 2 and name == "McNamara":
            action["status"]        = "🟠 校验 (Verify)"
            action["system_action"] = "输出倾向性结论，强制附加『需医生人工核对』标签"

            mc_label, mc_desc = _classify_McNamara(value)
            base = f"[{mc_label}] {mc_desc}。"

            if cvm_stage in ["CS1", "CS2", "CS3"]:
                action["clinical_advice"] = (
                    base + f"{cvm_d} 生长期上颌绝对突度参考，"
                    "强制核查 FH 平面稳定性（Porion SDR 68.7%，易被颞骨遮挡导致平面漂移）；"
                    "建议结合 ANB 综合判断前后向骨骼差异。"
                )
            else:
                action["clinical_advice"] = (
                    base + f"{cvm_d} 骨骼趋于定型，"
                    "McNamara 突距作为成人正畸/正颌手术设计参考，"
                    "需联合 ANB、Wits、软组织及 CBCT 综合分析，不可单独作为决策依据。"
                )
            return action

        # ── 其余第二层指标（U1_NA 等，value 可能由外部传入）──────
        if tier == 2:
            action["status"]          = "🟠 校验 (Verify)"
            action["system_action"]   = "输出倾向性结论，强制附加『需医生人工核对』标签"
            action["clinical_advice"] = f"{cvm_d} 阶段，{name} 数值 = {value}，请结合其他指标综合判断。"
            return action

        return action

    # ------------------------------------------------------------------
    def analyze(self, landmarks, cvm_stage="CS3", gender="F", pixel_spacing=0.084):
        lm = landmarks
        raw_values = {}

        try:
            raw_values["ANB"]      = calc_ANB(lm["S"], lm["N"], lm["A"], lm["B"])
            raw_values["Wits"]     = calc_Wits_FOP(lm["A"], lm["B"], lm["UPM"], lm["LPM"],
                                                    lm["UMT"], lm["LMT"], pixel_spacing)
            raw_values["E_line"]   = calc_E_line_upper(lm["Pn"], lm["Pog'"], lm["Ls"],
                                                        lm["N"], lm["S"], pixel_spacing)
            raw_values["McNamara"] = calc_McNamara_maxilla(lm["N"], lm["A"], lm["Po"],
                                                            lm["Or"], pixel_spacing)
        except Exception as e:
            print(f"测量指标计算异常: {e}")

        raw_values["FMA"] = None  # SDR 48.7%，直接进冻结逻辑

        final_report = {}
        for name, val in raw_values.items():
            if val is not None:
                val = round(val, 2)
            routing_res = self.run_cvm_routing(name, val, cvm_stage, raw_values, gender)
            final_report[name] = {
                "value":           val,
                "sdr":             self.SDR_TABLE.get(name, 0),
                "status":          routing_res["status"],
                "clinical_advice": routing_res["clinical_advice"],
                "system_action":   routing_res["system_action"],
            }
        return final_report


# ==========================================
# 4. 执行测试
# ==========================================
if __name__ == "__main__":
    demo_landmarks = {
        "S": [100.5, 80.2], "N": [105.0, 50.0],
        "A": [95.0, 120.0], "B": [90.0, 140.0],
        "Po":   [145.0, 68.0], "Or":  [135.0, 72.0],
        "Pn":   [82.0, 100.0], "Ls":  [87.0, 122.0], "Pog'": [87.0, 158.0],
        "UPM":  [105.0, 125.0], "LPM": [104.0, 127.0],
        "UMT":  [120.0, 122.0], "LMT": [119.0, 124.0],
    }

    engine = DiagnosticEngine()
    report = engine.analyze(demo_landmarks, cvm_stage="CS4", gender="F")

    print("\n" + "=" * 85)
    print(" 🏥 AI 专家级正畸诊断系统 - 最终输出报告 (CVM 阶段: CS4)")
    print("=" * 85)
    for indicator, data in report.items():
        val_str = f"{data['value']:>6.2f}" if data["value"] is not None else "   N/A"
        print(f"\n[{indicator}] (SDR: {data['sdr']}%)  数值: {val_str}")
        print(f"  权限状态: {data['status']}")
        print(f"  临床建议: {data['clinical_advice']}")
        print(f"  系统动作: {data['system_action']}")