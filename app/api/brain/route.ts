import { NextResponse } from 'next/server';
// 假设您使用的是 openai 官方库，它同样适用于 DeepSeek 等兼容 API
import OpenAI from 'openai';

// 初始化您的 AI 驱动引擎 (这里以 DeepSeek 为例，长官可换成其他的)
const openai = new OpenAI({
    apiKey: process.env.DEEPSEEK_API_KEY, 
    baseURL: 'https://api.deepseek.com/v1', // 如果用 OpenAI，删掉这行即可
});

export async function POST(req: Request) {
    try {
        const body = await req.json();
        const { command, sight, isSecurityOn } = body;

        // 🛡️ 战车可识别的目标字典（长官可以根据以后模型的能力扩充）
        const validTargets = ["person", "cell phone", "cup", "laptop", "bottle", "chair"];

        // 📜 核心军规：System Prompt
        const systemPrompt = `
你叫 JARVIS，是一个搭载在履带战车上的高智能 AI 视觉助理。
你目前的传感器状态如下：
- 视觉雷达当前看到的物体：[${sight || "未检测到物体"}]
- 安防监控模式：${isSecurityOn ? "已开启" : "已关闭"}

【你的任务】
分析用户的语音指令，并严格按照以下 JSON 格式返回，绝对不要输出任何 markdown 标记或其他多余的废话。
必须返回的 JSON 结构：
{
  "reply": "你简短、干练的语音回复（不超过15个字）",
  "action": "none" | "enable_security" | "disable_security" | "enable_hound" | "disable_hound",
  "target": "如果是开启猎犬模式，填入目标的英文名称，否则填 null"
}

【动作判断规则】
1. 如果用户让你“跟着那个人”、“追踪水杯”、“去咬那个瓶子”等：
   - action 设为 "enable_hound"
   - target 必须从这个列表里翻译并提取：[${validTargets.join(", ")}]。例如用户说“跟着那个人”，target 就是 "person"。
   - reply 示例："明白，猎犬已锁定目标！"
2. 如果用户让你“停下”、“别跟了”、“召回”：
   - action 设为 "disable_hound"
   - reply 示例："已召回，原地待命。"
3. 如果用户让你“开启安防”、“注意警戒”：
   - action 设为 "enable_security"
   - reply 示例："安防矩阵已激活。"
4. 如果用户让你“解除安防”、“安全了”：
   - action 设为 "disable_security"
5. 其他日常聊天或询问“你看到了什么”：
   - action 设为 "none"
   - reply 结合你当前看到的物体[${sight}]进行自然回复。
`;

        // 🚀 发送请求给云端大脑
        const response = await openai.chat.completions.create({
            model: "deepseek-chat", // 或者 gpt-4o-mini
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: command }
            ],
            // 强制 AI 返回纯 JSON 格式
            response_format: { type: "json_object" }, 
            temperature: 0.1, // 降低发散性，保证指令执行的稳定性
        });

        // 解析 AI 返回的结果
        const aiResponseText = response.choices[0].message.content;
        
        if (!aiResponseText) {
            throw new Error("AI 大脑未返回数据");
        }

        const aiDecision = JSON.parse(aiResponseText);
        
        // 打印到服务器控制台，方便长官监控
        console.log("🧠 AI 战术决策:", aiDecision);

        return NextResponse.json(aiDecision);

    } catch (error: any) {
        console.error("❌ 神经中枢短路:", error);
        return NextResponse.json(
            { reply: "报告，我的云端大脑似乎短路了。", action: "none", target: null },
            { status: 500 }
        );
    }
}