import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { imageBase64, message } = await request.json();
    
    // ⚠️ 1. 填入你在 Resend 刚刚复制的 API Key
    const RESEND_API_KEY = "re_jL5x4ere_Fbee4YDWUUx54LUqeadHeTMH"; 
    // ⚠️ 2. 填入你注册 Resend 时用的邮箱地址（免费测试阶段只能发给自己）
    const YOUR_EMAIL = "weblgy@163.com"; 

    // 构建非常专业的监控报警邮件
    const htmlContent = `
      <h2>🚨 AI 安防监控警报</h2>
      <p style="color: red; font-weight: bold; font-size: 18px;">${message}</p>
      <p>报警时间：${new Date().toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' })}</p>
      <hr />
      <p>现场抓拍截图：</p>
      <img src="${imageBase64}" alt="现场抓拍" style="max-width: 100%; border-radius: 8px; border: 2px solid red;" />
    `;

    // 调用 Resend 官方接口发送邮件
    const response = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Authorization": `Bearer ${RESEND_API_KEY}`
      },
      body: JSON.stringify({
        from: "Security Bot <onboarding@resend.dev>", // 这是 Resend 专门提供的免配置发送地址
        to: [YOUR_EMAIL],
        subject: "🚨 监控区域异常闯入！",
        html: htmlContent
      })
    });

    const result = await response.json();
    return NextResponse.json({ success: true, result });
    
  } catch (error) {
    console.error("邮件推送报错:", error);
    return NextResponse.json({ success: false }, { status: 500 });
  }
}