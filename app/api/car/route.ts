// 文件路径：app/api/car/route.ts
import { NextResponse } from 'next/server';
import dgram from 'dgram';

export async function POST(req: Request) {
    try {
        // 接收前端发来的指令和战车的 IP
        const { cmd, ip } = await req.json();
        
        if (!ip) {
            return NextResponse.json({ error: "Missing IP address" }, { status: 400 });
        }

        // 创建 UDP 客户端
        const client = dgram.createSocket('udp4');
        const message = Buffer.from(cmd);

        // 发送到 ESP32 的 8888 端口
        client.send(message, 0, message.length, 8888, ip, (err) => {
            if (err) console.error("UDP 发送失败:", err);
            client.close(); // 发送完毕立即释放端口
        });

        return NextResponse.json({ success: true, sent: cmd });
    } catch (error) {
        return NextResponse.json({ error: "Server error" }, { status: 500 });
    }
}