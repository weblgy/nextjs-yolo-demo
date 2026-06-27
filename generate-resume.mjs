import {
  Document, Packer, Paragraph, TextRun, HeadingLevel,
  Table, TableRow, TableCell, AlignmentType
} from 'docx';
import { writeFileSync } from 'fs';

// ===== Helper: simple bullet =====
function bullet(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 60 },
    indent: { left: 480, hanging: 240 },
    children: [
      new TextRun({ text: '•  ', size: 21, color: '2563eb' }),
      new TextRun({ text, size: 21, ...opts }),
    ],
  });
}

// ===== Helper: section heading =====
function sectionTitle(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 340, after: 140 },
    children: [
      new TextRun({
        text,
        size: 30,
        bold: true,
        color: '1e293b',
      }),
    ],
  });
}

// ===== Helper: project title + date =====
function projectHeader(title, date) {
  return new Paragraph({
    spacing: { before: 280, after: 100 },
    children: [
      new TextRun({ text: title, size: 27, bold: true, color: '1a56db' }),
      new TextRun({ text: `    ${date}`, size: 20, color: '94a3b8' }),
    ],
  });
}

// ===== Helper: tag chips =====
function tagRow(tags) {
  return new Paragraph({
    spacing: { after: 160 },
    children: tags.flatMap((t, i) => [
      new TextRun({ text: ` ${t} `, size: 18, color: '2563eb' }),
      ...(i < tags.length - 1 ? [new TextRun({ text: '  ', size: 18 })] : []),
    ]),
  });
}

// ===== Helper: sub-heading =====
function subTitle(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 80 },
    children: [{ text, size: 24, bold: true, color: '334155' }],
  });
}

// ===== Helper: body paragraph =====
function body(text) {
  return new Paragraph({
    spacing: { after: 80 },
    indent: { firstLine: 440 },
    children: [{ text, size: 21 }],
  });
}

// ===== Helper: difficulty table =====
function difficultyTable(rows) {
  return new Table({
    columnWidths: [3175, 5897],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            children: [new Paragraph({ children: [new TextRun({ text: '技术难点', size: 20, bold: true, color: '475569' })] })],
          }),
          new TableCell({
            children: [new Paragraph({ children: [new TextRun({ text: '解决方案', size: 20, bold: true, color: '475569' })] })],
          }),
        ],
      }),
      ...rows.map(row =>
        new TableRow({
          children: [
            new TableCell({
              children: [new Paragraph({ spacing: { before: 40, after: 40 }, children: [new TextRun({ text: row[0], size: 20, bold: true })] })],
            }),
            new TableCell({
              children: [new Paragraph({ spacing: { before: 40, after: 40 }, children: [new TextRun({ text: row[1], size: 20 })] })],
            }),
          ],
        })
      ),
    ],
  });
}

// ===== Build Document =====
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: 'Microsoft YaHei', size: 22, color: '333333' },
      },
    },
  },
  sections: [{
    children: [

      // ═══════ NAME ═══════
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 100 },
        children: [{ text: '李关宇', size: 54, bold: true, color: '1e293b' }],
      }),

      // ═══════ CONTACT ═══════
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: [
          new TextRun({ text: 'GitHub: github.com/weblgy', size: 20, color: '64748b' }),
          new TextRun({ text: '    |    ', size: 20, color: 'cbd5e1' }),
          new TextRun({ text: 'weblgy@163.com', size: 20, color: '64748b' }),
          new TextRun({ text: '    |    ', size: 20, color: 'cbd5e1' }),
          new TextRun({ text: '意向：AI应用开发  实习', size: 20, color: '2563eb' }),
        ],
      }),

      // ═══════ DIVIDER ═══════
      new Paragraph({ spacing: { after: 200 }, children: [] }),

      // ═══════ PERSONAL SUMMARY ═══════
      sectionTitle('个人概述'),
      body('热爱将 AI 模型从论文落地到实际场景的实践者，专注于大语言模型与计算机视觉的工程化应用。独立完成了一个完整的 AI 视觉推理终端——将 YOLO11 ONNX 模型部署至浏览器端边缘推理，集成 DeepSeek 大语言模型构建语音驱动的智能决策闭环，并拓展至 ESP32 嵌入式遥控子系统。熟练掌握 ONNX Runtime 边缘部署、大模型 Prompt Engineering、Agent 工具调用与结构化输出，同时具备 Next.js 全栈开发能力。善于解决实时系统中状态同步、UDP 协议通信、帧率控制等工程难题。'),

      // ═══════ SKILLS ═══════
      sectionTitle('技术能力'),
      skillTable(),

      // ═══════ PROJECTS ═══════
      sectionTitle('项目经历'),

      // --- Project 1 ---
      projectHeader('一、浏览器端 AI 视觉推理终端', '2025.05 – 2025.06'),
      tagRow(['Next.js 16', 'React 19', 'TypeScript', 'ONNX Runtime Web', 'YOLO11n/s', 'WebAssembly', 'Cache API']),

      subTitle('项目背景'),
      body('传统目标检测依赖服务端 GPU 推理，存在隐私泄漏风险与网络延迟瓶颈。本项目将 YOLO11 ONNX 模型完全部署在浏览器端，利用 WebAssembly SIMD 加速实现纯客户端推理——无需上传任何图片到服务器，延迟低且保护用户隐私。'),

      subTitle('核心职责'),
      bullet('模型工程化部署：将 5.6MB 的 YOLO11n ONNX 模型部署至浏览器端，通过 ONNX Runtime Web + WebAssembly SIMD 在消费级设备上实现每帧 15–30ms 推理速度'),
      bullet('多输入源适配：统一抽象摄像头实时流（getUserMedia）、静态图片上传、视频文件三种输入源，通过 requestAnimationFrame 驱动检测循环，100ms 帧率节流控制'),
      bullet('模型分发与缓存：利用浏览器 Cache API 实现模型文件本地持久化缓存，结合 ReadableStream 流式下载进度条，首次加载后实现"秒级装填"'),
      bullet('YOLO 后处理全链路自研：从零实现坐标反算（letterbox 逆向映射）、NMS 非极大值抑制（IoU 0.45）、Canvas 动态线宽自适应渲染、80 类 COCO 标签体系映射'),
      bullet('全栈 API 设计：基于 Next.js App Router 构建 3 条 API 路由（DeepSeek AI 大脑、UDP 车控中继、邮件告警推送），完成前后端完整闭环'),

      subTitle('技术难点与解决方案'),
      difficultyTable([
        ['浏览器 WASM 多线程兼容', '配置 numThreads=1 强制单核 + SIMD 开启，规避 SharedArrayBuffer 跨域策略限制'],
        ['rAF 循环中 React 状态闭包陈旧', '采用双轨状态架构：useState 驱动 UI 渲染 + useRef 同步写入关键业务变量，确保异步回调获取最新值'],
        ['letterbox 缩放坐标还原', '自研 preprocess() 实现保持宽高比的缩放+灰色填充，推理后精确反算坐标映射回原始画布'],
        ['跨分辨率自适应渲染', '动态线宽（max(imgWidth/100, 3)）与字号（max(imgWidth/30, 16)），适配不同输入尺寸'],
      ]),

      subTitle('优化成果'),
      bullet('摄像头分辨率降维至 640×480，单帧推理耗时从 ~80ms 降至 15–30ms'),
      bullet('每 5 帧更新一次推理时间 UI，避免高频 setState 造成重渲染抖动'),
      bullet('Canvas getContext 使用 willReadFrequently:true 标志优化频繁像素读取性能'),
      bullet('组件卸载时三层资源清理（cancelAnimationFrame + revokeObjectURL + track.stop），杜绝内存泄漏'),

      // --- Project 2 ---
      projectHeader('二、AI 语音驱动自动驾驶系统', '2025.06'),
      tagRow(['Next.js API Routes', 'Node.js dgram', 'DeepSeek Chat API', 'Web Speech API', 'ESP32', 'UDP 协议', 'PID 视觉伺服']),

      subTitle('项目背景'),
      body('对 ESP32 遥控车进行智能化改造，通过局域网 UDP 协议将 Web 控制台与嵌入式底盘互联。融合 YOLO 视觉检测 + DeepSeek 大语言模型 + Web Speech API，实现"一句话就让车跟着人走"的全自动语音驾驶体验。'),

      subTitle('核心职责'),
      bullet('UDP 指令中继网关：基于 Node.js dgram 模块设计无状态 UDP 透传——前端 POST → 服务端创建 Socket 发送 → 立即释放端口，零连接开销，延迟 < 10ms'),
      bullet('DeepSeek LLM 意图理解层：设计结构化 System Prompt（JSON 强制输出 + temperature=0.1），将中文语音指令映射为四项操作决策（追人/停下/开安防/关安防），结合当前 YOLO 视觉上下文实现情境感知'),
      bullet('视觉伺服 PID 追踪：基于检测框坐标设计拟态追踪算法——目标偏离中线 >40% 触发脉冲微调（50–80ms 精准转向），目标面积 >25% 触发自动泊车，丢失目标原地伏击等待'),
      bullet('AEB 自动紧急避障：实时评估前方障碍物（非目标物体占据画面中下走廊且面积 >30%），触发"后退+甩尾脱困"组合机动，1.5 秒冷却期防止决策震荡'),
      bullet('端到端语音闭环：Web Speech API 中文识别 → DeepSeek LLM 分析 → JSON 指令执行 → SpeechSynthesis TTS 播报，全链路延迟 < 2 秒'),

      subTitle('技术难点与解决方案'),
      difficultyTable([
        ['键盘按住不放导致 UDP 风暴', 'e.repeat 事件过滤 + keyup 自动刹车，每次仅发送一次指令，将发包密度从 60fps 降至 < 5fps'],
        ['窗口失焦导致车辆失控', '监听 window.blur 事件自动触发 Q 紧急制动，配合 fetch keepalive:true 确保失焦瞬间指令必达'],
        ['AI 决策与动作执行时序冲突', '引入 maneuverLockRef 机动冷却锁——指令发出后锁定 250ms–1.5s，禁止新决策覆盖正在执行的动作序列'],
        ['浏览器帧率抖动影响控制精度', '用 setTimeout 替代帧率依赖实现脉冲转向（50–80ms），回调自动发刹车指令，精确且帧率无关'],
      ]),

      // --- Project 3 ---
      projectHeader('三、智能安防监控与实时告警系统', '2025.06'),
      tagRow(['Next.js', 'Canvas 2D', 'Resend API', 'Web Speech API', 'Base64 编码', '电子围栏']),

      subTitle('项目背景'),
      body('在同一视觉终端上扩展安防监控功能——在摄像头画面中划定虚拟电子围栏，当检测到人员入侵时自动抓拍、邮件告警、并播放中文语音驱离警告，10 秒冷却机制防止告警轰炸。'),

      subTitle('核心职责'),
      bullet('虚拟电子围栏渲染：Canvas setLineDash 实时叠加红色虚线围栏（画面中央 60% 区域），视觉上清晰标记监控禁区'),
      bullet('入侵检测逻辑：遍历 YOLO 检测结果中 person 类目标，计算检测框中心点是否落入围栏区域（x: 20%–80%, y: 20%–80%）'),
      bullet('邮件告警推送链：Canvas 双图层合成（视频帧 + 检测框叠加）→ JPEG Base64 编码（质量 0.4）→ Resend API 发送带内嵌截图的 HTML 告警邮件'),
      bullet('TTS 语音驱离 + 10 秒冷却节流，防止持续入侵导致邮件轰炸'),

      // ═══════ EDUCATION ═══════
      sectionTitle('教育背景'),
      new Paragraph({
        spacing: { after: 40 },
        children: [
          new TextRun({ text: '本科 · 软件工程（或相关专业）', size: 22, bold: true }),
          new TextRun({ text: '    2024.09 – 2028.06（预计）', size: 20, color: '64748b' }),
        ],
      }),
      new Paragraph({
        spacing: { after: 200 },
        children: [{ text: '（请在此处填写学校名称，或按实际情况修改）', size: 20, color: '94a3b8', italics: true }],
      }),

      // ═══════ SELF ASSESSMENT ═══════
      sectionTitle('自我评价'),
      bullet('具备 AI 应用端到端交付能力——从 ONNX 模型选型与前端推理部署，到 LLM Prompt 工程设计与 Agent 决策闭环，均能独立完成'),
      bullet('理解 AI 模型特性与工程落地的边界——在 YOLO 精度/速度权衡、LLM temperature/结构化输出控制、WASM 多线程兼容等关键决策点上能做出务实的工程判断'),
      bullet('重视 AI 应用的用户体验链路——语音识别→LLM 理解→动作执行→TTS 播报的端到端延迟控制在 2 秒以内，关注交互流畅性'),
      bullet('具备跨技术栈的广度优势——从浏览器端 AI 推理延伸到 ESP32 嵌入式控制，能以系统工程视角看待 AI 应用的全链路技术问题'),
    ],
  }],
});

// ===== Skill Table (simplified, no shading) =====
function skillTable() {
  const skills = [
    ['AI / 模型推理', 'ONNX Runtime Web, YOLO11, ONNX 模型部署与优化, WebAssembly SIMD 加速'],
    ['LLM / AI Agent', 'DeepSeek Chat API, OpenAI SDK, Prompt Engineering, JSON 结构化输出, Agent 工具调用'],
    ['Web 全栈', 'Next.js 16, React 19, TypeScript, Next.js API Routes, HTTP/REST, Resend API'],
    ['Web API / 媒体', 'Web Speech API (STT/TTS), Cache API, Canvas 2D, MediaStream, ReadableStream'],
    ['嵌入式/IoT', 'ESP32, UDP 局域网通信, 遥控车底盘控制, PID 视觉伺服'],
    ['工程化', 'Git, npm, Tailwind CSS v4, shadcn/ui, ESLint, 资源生命周期管理'],
  ];

  return new Table({
    columnWidths: [2000, 7072],
    rows: skills.map(([label, content]) =>
      new TableRow({
        children: [
          new TableCell({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: label, size: 20, bold: true, color: '2563eb' })],
            })],
          }),
          new TableCell({
            children: [new Paragraph({
              children: [new TextRun({ text: content, size: 20 })],
            })],
          }),
        ],
      })
    ),
  });
}

// ===== Generate =====
const desktop = 'C:\\Users\\Administrator\\Desktop';
const buffer = await Packer.toBuffer(doc);
const now = new Date();
const ts = `${now.getMonth()+1}${now.getDate()}_${now.getHours()}${now.getMinutes()}`;
const outPath = `${desktop}\\李关宇_AI应用开发实习_简历.docx`;
try {
  writeFileSync(outPath, buffer);
  console.log('✅ 简历已生成：李关宇_AI应用开发实习_简历.docx');
} catch (e) {
  const alt = `${desktop}\\李关宇_AI应用开发实习_简历_${ts}.docx`;
  writeFileSync(alt, buffer);
  console.log(`原文件被占用，已生成新版本：李关宇_AI应用开发实习_简历_${ts}.docx`);
}
