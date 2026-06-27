# 🧠 JARVIS Vision Terminal

> **AI 应用工程全链路作品** — 从浏览器端 ONNX 模型推理、LLM Agent 决策、到嵌入式硬件控制，展示 AI 应用开发的核心能力。

[![Tech Stack](https://img.shields.io/badge/AI-ONNX%20Runtime%20Web-blue)](https://onnxruntime.ai/)
[![LLM](https://img.shields.io/badge/LLM-DeepSeek%20Chat-purple)](https://platform.deepseek.com/)
[![Framework](https://img.shields.io/badge/Framework-Next.js%2016-black)](https://nextjs.org/)
[![Model](https://img.shields.io/badge/Model-YOLO11-orange)](https://docs.ultralytics.com/)

---

## 📖 项目简介

传统目标检测依赖服务端 GPU 推理，存在**隐私泄漏**与**网络延迟**双重瓶颈。本项目将 YOLO11 ONNX 模型完全部署在浏览器端，利用 **WebAssembly SIMD** 实现纯客户端推理——图片绝不离开你的设备。

在此基础上，集成 **DeepSeek 大语言模型**构建语音驱动的智能决策闭环，并通过 **UDP 协议**连接 ESP32 嵌入式遥控车底盘，实现"一句话就让车跟着人走"的全自动 AI 驾驶体验。

```
┌──────────────────────────────────────────────────────────┐
│                    浏览器端 (Browser)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ 摄像头 📷  │  │ 麦克风 🎤  │  │ YOLO11 ONNX 推理 🧠  │   │
│  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘   │
│       │             │                   │                │
│       ▼             ▼                   ▼                │
│  ┌──────────────────────────────────────────────────┐    │
│  │           多模态 AI Agent 决策引擎                  │    │
│  │  视觉检测 + 语音指令 → DeepSeek LLM → 结构化动作    │    │
│  └──────────────────────┬───────────────────────────┘    │
└─────────────────────────┼────────────────────────────────┘
                          │ HTTP POST
                          ▼
┌──────────────────────────────────────────────────────────┐
│                  Next.js API Routes (服务端)               │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ /api/brain   │  │  /api/car    │  │ /api/wechat    │  │
│  │ LLM 决策 🧠  │  │ UDP 车控 🚗  │  │ 邮件告警 📧    │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘  │
└─────────┼─────────────────┼──────────────────┼───────────┘
          │                 │ UDP :8888        │ Resend API
          ▼                 ▼                  ▼
   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
   │ DeepSeek API │  │ ESP32 战车 🏎️ │  │  你的邮箱 📬  │
   └─────────────┘  └──────────────┘  └──────────────┘
```

---

## 🎯 核心能力展示

### 1. AI 模型边缘部署 (Edge Inference)

- 将 **YOLO11 ONNX 模型**（5.6MB）部署至浏览器端，利用 **ONNX Runtime Web + WebAssembly SIMD** 实现纯客户端推理
- 单帧推理 **15–30ms**，无需服务端 GPU，数据不出浏览器
- 自研 YOLO 后处理全链路：**letterbox 坐标反算 → NMS 非极大值抑制 → Canvas 动态渲染**
- 模型分发与缓存：**Cache API** 本地持久化 + **ReadableStream** 流式下载进度条

### 2. LLM Agent 决策系统

- 集成 **DeepSeek Chat API**（兼容 OpenAI SDK）
- 设计结构化 **System Prompt**：角色设定 + 视觉上下文注入 + JSON 强制输出
- 关键参数控制：`temperature: 0.1`（稳定性）+ `response_format: json_object`（结构化）
- 自然语言 → 结构化动作映射：追人/停下/开安防/关安防

### 3. 多模态交互闭环

- **Web Speech API** 中文语音识别（STT）+ 语音合成（TTS）
- 视觉 + 语音 + 动作的完整闭环，端到端延迟 **< 2 秒**
- 语音指令示例："跟着那个人"、"开启安防"、"停下来"

### 4. IoT 硬件控制

- **Node.js dgram** 实现 UDP 局域网透传，无状态设计，零连接开销
- ESP32 遥控车 **PID 视觉伺服追踪** + **AEB 自动紧急避障**
- 窗口失焦保护、按键防抖、机动冷却锁等工程安全机制

### 5. 智能安防监控

- Canvas 虚拟电子围栏实时渲染
- 入侵检测 → **Canvas 双图层合成抓拍** → JPEG Base64 编码 → **Resend API** 邮件告警
- TTS 语音驱离 + 10 秒冷却节流

---

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| **AI 推理** | ONNX Runtime Web, YOLO11n/s, WebAssembly SIMD |
| **LLM / Agent** | DeepSeek Chat API, OpenAI SDK, Prompt Engineering, JSON 结构化输出 |
| **前端框架** | Next.js 16 (App Router), React 19, TypeScript |
| **样式** | Tailwind CSS v4, shadcn/ui (Radix UI), Lucide Icons |
| **后端 API** | Next.js API Routes, Node.js dgram (UDP Socket) |
| **Web API** | Web Speech API, Cache API, Canvas 2D, MediaStream, ReadableStream |
| **硬件/IoT** | ESP32, UDP 局域网通信协议 |
| **第三方服务** | Resend (邮件), DeepSeek (LLM) |
| **工程化** | ESLint, TypeScript, 资源生命周期管理 |

---

## 🚀 快速启动

### 环境要求

- Node.js 18+
- npm / yarn / pnpm

### 1. 安装依赖

```bash
npm install
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
# 获取地址: https://platform.deepseek.com/api_keys
```

### 3. 下载 ONNX 模型文件

将 YOLO11 ONNX 模型文件放入 `public/model/` 目录：

```
public/model/
├── yolo11n.onnx    # YOLO11 Nano (推荐，5.6MB)
└── yolo11s.onnx    # YOLO11 Small (可选)
```

> 模型可从 [Ultralytics](https://docs.ultralytics.com/models/yolo11/) 导出为 ONNX 格式。

### 4. 启动开发服务器

```bash
npm run dev
```

打开 [http://localhost:3000](http://localhost:3000) 查看效果，访问 `/demo` 进入视觉推理终端。

---

## 📂 项目结构

```
my-ai-portfolio/
├── app/
│   ├── layout.tsx              # 根布局
│   ├── page.tsx                # 首页
│   ├── globals.css             # 全局样式（Tailwind）
│   ├── demo/
│   │   └── page.tsx            # 🔥 核心 Demo：视觉推理终端
│   └── api/
│       ├── brain/route.ts      # 🧠 DeepSeek LLM AI 决策 API
│       ├── car/route.ts        # 🚗 UDP 车控中继网关
│       └── wechat/route.ts     # 📧 安防邮件告警推送
├── components/
│   ├── Navbar.tsx              # 导航栏
│   └── ui/                     # shadcn/ui 组件
├── lib/
│   └── utils.ts                # YOLO 预处理/后处理工具函数
├── public/
│   ├── model/                  # ONNX 模型文件
│   │   ├── yolo11n.onnx
│   │   └── yolo11s.onnx
│   └── wasm/                   # ONNX Runtime WASM 运行时
├── generate-resume.mjs          # 简历生成脚本
├── .env.example                # 环境变量模板
└── package.json
```

---

## 🎯 Prompt Engineering 设计

本项目使用结构化 System Prompt 将 LLM 嵌入 Agent 决策流水线：

```typescript
// 核心设计要素（详见 app/api/brain/route.ts）

// 1. 角色设定 — 定义 Agent 行为边界
"你叫 JARVIS，是一个搭载在履带战车上的高智能 AI 视觉助理"

// 2. 上下文注入 — 动态注入实时视觉数据
"视觉雷达当前看到的物体：[${sight}]"
"安防监控模式：${isSecurityOn ? '已开启' : '已关闭'}"

// 3. 结构化约束 — 确保输出可被程序消费
response_format: { type: "json_object" }  // 强制 JSON 输出
temperature: 0.1                           // 降低发散性

// 4. 动作映射 — 自然语言 → 可执行指令
{
  "reply": "明白，猎犬已锁定目标！",
  "action": "enable_hound",
  "target": "person"
}
```

---

## 🔧 解决的关键工程问题

| 挑战 | 解决方案 |
|------|----------|
| 浏览器 WASM 多线程兼容 | 配置 `numThreads=1` 强制单核 + SIMD 开启，规避 SharedArrayBuffer 跨域策略 |
| rAF 循环中 React 闭包陈旧 | **双轨状态架构**：useState 驱动 UI + useRef 同步关键业务变量 |
| YOLO letterbox 坐标还原 | 自研 preprocess() 保持宽高比缩放 + 灰色填充 + 精确反算 |
| 键盘按住导致 UDP 风暴 | `e.repeat` 事件过滤 + keyup 自动刹车，发包密度 60fps → <5fps |
| 窗口失焦导致车辆失控 | `window.blur` 事件自动触发 Q 紧急制动 + `fetch keepalive: true` |
| AI 决策覆盖正在执行的动作 | `maneuverLockRef` 机动冷却锁 — 指令发出后锁定 250ms–1.5s |
| 持续入侵导致邮件轰炸 | 10 秒冷却节流机制 |

---

## 📝 技术文章

本项目涉及的核心技术点，推荐延伸阅读：

- [ONNX Runtime Web 官方文档](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [YOLO11 模型文档](https://docs.ultralytics.com/models/yolo11/)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
- [WebAssembly SIMD 介绍](https://v8.dev/features/simd)

---

## 📄 License

MIT

---

> 🤖 本项目为个人 AI 应用开发能力展示作品。作者：李关宇 | [GitHub](https://github.com/weblgy)
