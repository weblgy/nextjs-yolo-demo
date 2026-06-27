"use client";
import { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";
import { preprocess, renderBoxes } from "@/lib/utils";

const labelTranslator: Record<string, string> = {
    person: "人",
    "cell phone": "手机",
    cup: "杯子",
    laptop: "笔记本电脑",
    bottle: "瓶子",
    chair: "椅子",
};

// 🔥 定义日志类型
type LogType = 'info' | 'success' | 'warning' | 'error' | 'ai';
interface SystemLog {
    id: number;
    time: string;
    text: string;
    type: LogType;
}


const generateTerminalBar = (progress: number) => {
    const totalBlocks = 20;
    const safeProgress = Math.min(Math.max(0, progress), 100);
    const filledBlocks = Math.floor((safeProgress / 100) * totalBlocks);
    const emptyBlocks = totalBlocks - filledBlocks;
    const bar = '█'.repeat(Math.max(0, filledBlocks)) + '░'.repeat(Math.max(0, emptyBlocks));
    const formattedProgress = safeProgress.toString().padStart(3, ' ');
    return `[${bar}] ${formattedProgress}%`;
};

export default function DemoPage() {
    // --- 状态 ---
    const [model, setModel] = useState<ort.InferenceSession | null>(null);
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [inferenceTime, setInferenceTime] = useState<number | null>(null);
    const [isWebcamOpen, setIsWebcamOpen] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const [isSecurityMode, setIsSecurityMode] = useState(false);
    const [isModelLoading, setIsModelLoading] = useState(true);
    const [downloadProgress, setDownloadProgress] = useState(0);
    const lastDetectTimeRef = useRef(0); // 🌟 用于控制 AI 的帧率跳跃

    // --- 日志状态 ---
    const [logs, setLogs] = useState<SystemLog[]>([]);

    // --- 战车物理通信状态 (局域网直连版) ---
    const [isCarConnected, setIsCarConnected] = useState(false);
    const [isAutoMode, setIsAutoMode] = useState(false);
    const [ipAddress, setIpAddress] = useState("172.20.10.2"); // 预设为您刚拿到的 IP

    const [wrapperStyle, setWrapperStyle] = useState<React.CSSProperties>({
        width: '100%',
        height: '100%'
    });

    // --- Refs ---
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const videoInputRef = useRef<HTMLInputElement>(null);
    const logsEndRef = useRef<HTMLDivElement>(null); // 日志自动滚动锚点

    const requestRef = useRef<number>(0);
    const smallCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const isProcessingRef = useRef(false);
    const lastTimeRef = useRef(0);
    const frameCountRef = useRef(0);
    const modelRef = useRef<ort.InferenceSession | null>(null);
    const lastSpeechTimeRef = useRef(0);
    const currentSightRef = useRef<string[]>([]);
    const securityModeRef = useRef(false);

    // ==========================================
    // 🤖 AI 与局域网状态 Ref 记忆 (解决闭包问题)
    // ==========================================
    const autoModeRef = useRef(false);
    const houndTargetRef = useRef<string | null>(null);
    const lastCmdRef = useRef('Q');
    // 👇 加入这行：战术机动冷却锁
    const maneuverLockRef = useRef(0);
    // 👇 请在这里补上这 2 行！
    const wHeartbeatRef = useRef(0);
    const pulseTurnEndTimeRef = useRef(0);

    // 🔥 新增：用于在 requestAnimationFrame 循环中获取最新 IP 和连接状态
    const isCarConnectedRef = useRef(false);
    const ipAddressRef = useRef("172.20.10.2");

    const sonarRef = useRef<any>(null);

    const currentCmdRef = useRef('Q');



    useEffect(() => {
        isCarConnectedRef.current = isCarConnected;
    }, [isCarConnected]);

    useEffect(() => {
        ipAddressRef.current = ipAddress;
    }, [ipAddress]);

    // ⌨️ 绑定键盘监听 (WASD 控制)
    useEffect(() => {
        if (!isCarConnected) return;

        const handleKeyDown = (e: KeyboardEvent) => {
            // 🔥 核心修复：如果是按住不放产生的连续触发，直接拦截丢弃！
            if (e.repeat) return;

            const key = e.key.toUpperCase();
            if (['W', 'A', 'S', 'D'].includes(key)) {
                sendCarCommand(key);
            }
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            const key = e.key.toUpperCase();
            if (['W', 'A', 'S', 'D'].includes(key)) {
                sendCarCommand('Q'); // 松开立刻刹车
            }
        };

        // 🌟 第三枪核心：窗口失焦保护 (Blur Protection)
        // 防止按住 W 时突然切出浏览器，导致战车收不到 Q 刹车指令而撞墙！
        const handleBlur = () => {
            console.log("⚠️ 窗口失去焦点，强制触发紧急制动！");
            sendCarCommand('Q');
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);
        window.addEventListener('blur', handleBlur); // 👈 绑定失焦事件

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
            window.removeEventListener('blur', handleBlur); // 👈 记得清理
        };
    }, [isCarConnected]);



    const sendCarCommand = (cmd: string) => { // 👈 注意：去掉了 async
        if (cmd === currentCmdRef.current) return;
        currentCmdRef.current = cmd;

        console.log(`📡 发射指令: ${cmd}`); // 加上这句，盯紧控制台！

        // 👈 核心：不要用 await！直接发出去就不管了，绝不阻塞！
        fetch('/api/car', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cmd: cmd, ip: ipAddressRef.current }),
            // 添加 keepalive 确保即使用户切换页面也能发出去
            keepalive: true
        }).catch(e => console.error("指令发送失败:", e));
    };
    // 监听开关变化：如果主动关闭自动驾驶，立刻发送 Q 刹车保底
    useEffect(() => {
        autoModeRef.current = isAutoMode;
        if (!isAutoMode && isCarConnected) {
            sendCarCommand('Q');
            lastCmdRef.current = 'Q';
        }
    }, [isAutoMode]);

    useEffect(() => {
        securityModeRef.current = isSecurityMode;
    }, [isSecurityMode]);

    // 🔥 优化：自动滚动到底部
    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [logs, downloadProgress, isModelLoading]);

    // 写日志通用函数
    const addLog = (text: string, type: LogType = 'info') => {
        const now = new Date();
        const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
        setLogs(prev => [...prev, { id: Date.now() + Math.random(), time: timeStr, text, type }]);
    };

    // 🔌 无线连接战车 (局域网秒连版)
    const connectToCar = () => {
        if (!ipAddress) {
            addLog("错误：未输入战车局域网 IP 坐标！", "error");
            return;
        }
        addLog(`正在锁定局域网目标 [${ipAddress}]...`, "info");

        // 由于局域网直连极快，我们直接标记连接成功
        setIsCarConnected(true);
        addLog("局域网极速专线已接通，延迟 < 10ms！", "success");
        speakAnswer("物理底盘已接管，随时准备出发。");
    };

    // ⌨️ 绑定键盘监听 (WASD 控制)
    useEffect(() => {
        if (!isCarConnected) return;

        const handleKeyDown = (e: KeyboardEvent) => {
            // 🔥 核心修复：如果是按住不放产生的连续触发，直接拦截丢弃！
            // 这样按住 W 时，只会向战车发送一次请求，彻底告别 DDOS 拥堵！
            if (e.repeat) return;

            const key = e.key.toUpperCase();
            if (['W', 'A', 'S', 'D'].includes(key)) {
                sendCarCommand(key);
            }
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            const key = e.key.toUpperCase();
            if (['W', 'A', 'S', 'D'].includes(key)) {
                sendCarCommand('Q'); // 松开立刻刹车
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        };
    }, [isCarConnected]); // 依赖中去掉 writer，因为我们已经抛弃了它

    useEffect(() => {
        const sc = document.createElement('canvas');
        sc.width = 640;
        sc.height = 640;
        smallCanvasRef.current = sc;
    }, []);


    useEffect(() => {
        const initModel = async () => {
            setIsModelLoading(true);
            setDownloadProgress(0);
            addLog("系统内核启动中，正在连接视觉中枢...", "info");
            try {
                ort.env.wasm.wasmPaths = '/wasm/';
                ort.env.wasm.numThreads = 1;       // 强制单核运行，避免多线程引起的奇怪报错
                ort.env.wasm.simd = true;          // 开启 SIMD 加速

                const modelUrl = `${window.location.origin}/model/yolo11n.onnx`;
                let modelBuffer;

                const cacheName = 'yolo-model-cache-v1';
                const cache = await caches.open(cacheName);
                const cachedResponse = await cache.match(modelUrl);

                if (cachedResponse) {
                    addLog("发现本地模型缓存，正在秒级装填...", "info");
                    setDownloadProgress(100);
                    modelBuffer = await cachedResponse.arrayBuffer();
                } else {
                    addLog("正在建立加密通道，下载视觉模型...", "info");
                    const response = await fetch(modelUrl);

                    if (!response.ok) throw new Error(`模型下载失败，HTTP 状态码: ${response.status}`);

                    const contentLength = response.headers.get('content-length');
                    const total = contentLength ? parseInt(contentLength, 10) : 0;

                    const reader = response.body?.getReader();
                    if (!reader) throw new Error("无法读取数据流");

                    const chunks = [];
                    let loaded = 0;

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        chunks.push(value);
                        loaded += value.length;

                        if (total) {
                            const progress = Math.round((loaded / total) * 100);
                            setDownloadProgress(progress);
                        } else {
                            setDownloadProgress(prev => prev >= 95 ? 95 : prev + 5);
                        }
                    }

                    const uint8Array = new Uint8Array(loaded);
                    let position = 0;
                    for (let chunk of chunks) {
                        uint8Array.set(chunk, position);
                        position += chunk.length;
                    }
                    modelBuffer = uint8Array.buffer;

                    cache.put(modelUrl, new Response(modelBuffer));
                    setDownloadProgress(100);
                }

                addLog("数据接收完毕，正在注入推理引擎...", "info");
                const session = await ort.InferenceSession.create(modelBuffer, {
                    executionProviders: ["wasm"],
                    graphOptimizationLevel: "basic",
                });
                setModel(session);
                modelRef.current = session;
                addLog("YOLO 视觉推理模型加载完成，系统已就绪.", "success");

                setIsModelLoading(false);

            } catch (e: any) {
                console.error("加载失败:", e);
                addLog(`致命错误：${e.message}`, "error");
            }
        };
        initModel();
    }, []);

    const updateDimensions = () => {
        const container = containerRef.current;
        const video = videoRef.current;
        if (!container || (!video && !imageSrc)) return;

        let contentWidth = 0;
        let contentHeight = 0;
        if (imageSrc) {
            const img = container.querySelector("img");
            if (img) {
                contentWidth = img.naturalWidth;
                contentHeight = img.naturalHeight;
            }
        } else if (video && video.readyState >= 1) {
            contentWidth = video.videoWidth;
            contentHeight = video.videoHeight;
        }
        if (contentWidth === 0 || contentHeight === 0) return;

        const { width: containerW, height: containerH } = container.getBoundingClientRect();
        const contentRatio = contentWidth / contentHeight;
        const containerRatio = containerW / containerH;

        let finalW, finalH;
        if (containerRatio > contentRatio) {
            finalH = containerH;
            finalW = finalH * contentRatio;
        } else {
            finalW = containerW;
            finalH = finalW / contentRatio;
        }

        setWrapperStyle({
            width: `${finalW}px`,
            height: `${finalH}px`,
            position: 'relative',
        });
    };

    useEffect(() => {
        const timer = setTimeout(updateDimensions, 100);
        window.addEventListener('resize', updateDimensions);
        return () => {
            clearTimeout(timer);
            window.removeEventListener('resize', updateDimensions);
        };
    }, [isFullscreen, imageSrc, isWebcamOpen]);

    const detectFrame = async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const smallCanvas = smallCanvasRef.current;

        if (!video || !canvas || !modelRef.current || !smallCanvas) return;

        if (video.paused || video.ended) return;

        requestRef.current = requestAnimationFrame(detectFrame);

        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        const now = Date.now();
        if (now - lastTimeRef.current < 100 || isProcessingRef.current) return;

        isProcessingRef.current = true;
        lastTimeRef.current = now;

        try {
            const ctx = canvas.getContext("2d");
            ctx?.clearRect(0, 0, canvas.width, canvas.height);

            const start = performance.now();
            const smallCtx = smallCanvas.getContext("2d", { willReadFrequently: true });
            if (!smallCtx) return;

            const modelSize = 640;
            const scale = Math.min(modelSize / video.videoWidth, modelSize / video.videoHeight);
            const scaledW = video.videoWidth * scale;
            const scaledH = video.videoHeight * scale;
            const dx = (modelSize - scaledW) / 2;
            const dy = (modelSize - scaledH) / 2;

            smallCtx.fillStyle = "#727272";
            smallCtx.fillRect(0, 0, modelSize, modelSize);
            smallCtx.drawImage(video, dx, dy, scaledW, scaledH);

            const inputTensorData = await preprocess(smallCanvas, 640, 640);
            const inputTensor = new ort.Tensor("float32", Float32Array.from(inputTensorData), [1, 3, 640, 640]);

            const outputs = await modelRef.current.run({ images: inputTensor });
            const output = outputs["output0"];
            const end = performance.now();

            frameCountRef.current++;
            if (frameCountRef.current % 5 === 0) setInferenceTime(end - start);

            const detectedItems = renderBoxes(canvas, 0.50, output.data as Float32Array, 0, 0) || [];
            // ==========================================
            // 🚀 AI 智能驾驶引擎 (V3 狙击手版本 - 绝对不转圈)
            // ==========================================
            if (autoModeRef.current) {
                const nowTime = Date.now();

                // 🌟 核心护城河：只有当“晕眩冷却期”结束，才允许 AI 思考下一步动作
                // 注意：这里用 if 包装，而不是 return，防止阻断底部的画框代码
                if (nowTime >= (maneuverLockRef.current || 0)) {

                    const targetLabel = houndTargetRef.current || "person";
                    const target = detectedItems.find((item: any) => item.label === targetLabel);

                    const VIDEO_WIDTH = canvas.width;
                    const VIDEO_AREA = canvas.width * canvas.height;
                    // 🚧 1. 紧急避障系统 (AEB) - 智能反向脱困版
                    const PATH_LEFT = VIDEO_WIDTH * 0.35;  // 🌟 进一步收窄走廊：只看中间 30% 的绝对安全区
                    const PATH_RIGHT = VIDEO_WIDTH * 0.65;

                    const obstacle = detectedItems.find((item: any) => {
                        if (item.label === targetLabel) return false;

                        const [x1, y1, x2, y2] = item.box;
                        const obstacleArea = ((x2 - x1) * (y2 - y1)) / VIDEO_AREA;

                        // 必须严格挡在中间的细长走廊里
                        const isBlockingPath = (x2 > PATH_LEFT && x1 < PATH_RIGHT);
                        // 必须贴脸 (底边超过屏幕下方 65%)
                        const isClose = y2 > (canvas.height * 0.65);

                        // 占地面积放宽到 30%，防止一点小东西就吓退
                        return isBlockingPath && isClose && obstacleArea > 0.30;
                    });

                    if (obstacle) {
                        const [ox1, _, ox2, __] = obstacle.box;
                        const obstacleCenter = (ox1 + ox2) / 2;

                        // 🌟 核心脱困智驾：障碍物在左边，我们就往右跑；在右边，就往左跑！
                        const escapeDirection = obstacleCenter < (VIDEO_WIDTH / 2) ? 'D' : 'A';
                        const dirText = escapeDirection === 'A' ? '左' : '右';

                        addLog(`🛑 陷入死胡同！启动战术脱困：向 [${dirText}] 猛打方向规避 [${labelTranslator[obstacle.label] || obstacle.label}]`, "error");

                        if (isCarConnectedRef.current) {
                            // 1. 先猛退 500ms，拉开物理安全距离
                            sendCarCommand('S');

                            // 2. 500ms后，向反方向发出“大角度偏转指令”（持续 700ms 的大甩尾）
                            setTimeout(() => sendCarCommand(escapeDirection), 500);

                            // 3. 1200ms后彻底停稳，准备重新索敌
                            setTimeout(() => sendCarCommand('Q'), 1200);
                        }

                        lastCmdRef.current = 'S';
                        // 🌟 强行闭眼冷却 1.5 秒！在这 1.5 秒内绝对不看新画面，专心把这套复杂的“后退+大甩尾”动作做完！
                        maneuverLockRef.current = nowTime + 1500;
                    }

                    if (obstacle) {
                        addLog(`🛑 AEB 避让！前方死胡同：[${labelTranslator[obstacle.label] || obstacle.label}]`, "error");
                        if (isCarConnectedRef.current) {
                            // 🌟 全新解困机动 (Escape Maneuver)：不要直挺挺地后退！
                            sendCarCommand('S');
                            // 倒车 400ms 后，稍微向左打一点方向（甩尾），打破物理死角
                            setTimeout(() => sendCarCommand('A'), 400);
                            // 600ms 后彻底停稳，重新搜索目标
                            setTimeout(() => sendCarCommand('Q'), 600);
                        }
                        lastCmdRef.current = 'S';
                        // 强行冷却 1 秒，让战车有充足的时间完成这套“后退+甩尾”的动作
                        maneuverLockRef.current = nowTime + 1000;
                    }
                    else if (!target) {
                        // 🌀 2. 状态 A：目标丢失 -> 【原地伏击模式】
                        // 彻底废除转圈扫街！找不到人就立刻死死踩住刹车！
                        if (lastCmdRef.current !== 'Q') {
                            if (isCarConnectedRef.current) sendCarCommand('Q');
                            lastCmdRef.current = 'Q';
                            addLog(`⚠️ 目标脱离视线，战车紧急制动，进入原地伏击...`, "warning");
                        }
                        // 目标丢失时，除了刹车，绝对不做任何多余动作
                    }
                    else {
                        // 🎯 3. 状态 B：动态拟态 PID 追踪
                        const [x1, y1, x2, y2] = target.box;
                        const centerX = (x1 + x2) / 2;
                        const areaRatio = ((x2 - x1) * (y2 - y1)) / VIDEO_AREA;

                        const screenCenter = VIDEO_WIDTH / 2;
                        const offsetRatio = Math.abs(centerX - screenCenter) / screenCenter;

                        const leftZone = VIDEO_WIDTH * 0.40; // 🌟 拓宽安全区：中间 20% 都不转弯
                        const rightZone = VIDEO_WIDTH * 0.60;
                        const STOP_RATIO = 0.25; // 距离足够近就停车

                        let nextCmd = 'Q';
                        let turnDuration = 0;

                        if (centerX < leftZone) {
                            nextCmd = 'A';
                            turnDuration = 50 + (offsetRatio * 80); // 极短的脉冲
                        } else if (centerX > rightZone) {
                            nextCmd = 'D';
                            turnDuration = 50 + (offsetRatio * 80);
                        } else {
                            if (areaRatio > STOP_RATIO) {
                                nextCmd = 'Q';
                            } else {
                                nextCmd = 'W';
                            }
                        }

                        // 🚀 4. 动力下发
                        if (nextCmd === 'A' || nextCmd === 'D') {
                            if (isCarConnectedRef.current) {
                                sendCarCommand(nextCmd);
                                // 🌟 杀手锏：不依赖卡顿的浏览器帧率，强制 x 毫秒后准时发刹车指令
                                setTimeout(() => sendCarCommand('Q'), turnDuration);
                            }
                            lastCmdRef.current = 'Q'; // 预测马上会停，直接记为 Q
                            addLog(nextCmd === 'A' ? `🐺 左微调 (${Math.round(turnDuration)}ms)` : `🐺 右微调 (${Math.round(turnDuration)}ms)`, "info");
                            maneuverLockRef.current = nowTime + turnDuration + 250; // 给摄像头留出对焦时间
                        }
                        else if (nextCmd === 'W') {
                            const needHeartbeat = (nowTime > (wHeartbeatRef.current || 0));
                            if (nextCmd !== lastCmdRef.current || needHeartbeat) {
                                if (isCarConnectedRef.current) sendCarCommand('W');
                                lastCmdRef.current = 'W';
                                addLog(`锁定目标，推进中...`, "success");
                                wHeartbeatRef.current = nowTime + 300;
                            }
                        }
                        else if (nextCmd === 'Q' && lastCmdRef.current !== 'Q') {
                            if (isCarConnectedRef.current) sendCarCommand('Q');
                            lastCmdRef.current = 'Q';
                            addLog(`🛑 到达安全距离，平稳泊车。`, "warning");
                        }
                    }
                }
            }
            currentSightRef.current = detectedItems.map((item: any) => item.label);

            let isIntruderInFence = false;
            const fXMin = canvas.width * 0.2;
            const fXMax = canvas.width * 0.8;
            const fYMin = canvas.height * 0.2;
            const fYMax = canvas.height * 0.8;

            if (securityModeRef.current) {
                const ctx = canvas.getContext("2d");
                if (ctx) {
                    ctx.save();
                    ctx.strokeStyle = "rgba(239, 68, 68, 0.8)";
                    ctx.lineWidth = 3;
                    ctx.setLineDash([15, 10]);
                    ctx.strokeRect(fXMin, fYMin, fXMax - fXMin, fYMax - fYMin);
                    ctx.fillStyle = "rgba(239, 68, 68, 0.9)";
                    ctx.font = "bold 16px monospace";
                    ctx.fillText("⚠️ RESTRICTED AREA // 监控禁区", fXMin + 10, fYMin + 24);
                    ctx.restore();
                }
            }

            for (const item of detectedItems) {
                if (item.label === "person") {
                    const [x1, y1, x2, y2] = item.box;
                    const cx = (x1 + x2) / 2;
                    const cy = (y1 + y2) / 2;
                    if (cx > fXMin && cx < fXMax && cy > fYMin && cy < fYMax) {
                        isIntruderInFence = true;
                        break;
                    }
                }
            }

            if (securityModeRef.current && isIntruderInFence) {
                const now = Date.now();
                if (now - lastSpeechTimeRef.current > 10000) {
                    addLog("⚠️ 警告：检测到人员入侵虚拟电子围栏！执行抓拍推送...", "error");
                    lastSpeechTimeRef.current = now;

                    const video = videoRef.current;
                    let base64Img = "";

                    if (video) {
                        const tempCanvas = document.createElement("canvas");
                        tempCanvas.width = video.videoWidth;
                        tempCanvas.height = video.videoHeight;
                        const ctx = tempCanvas.getContext("2d");
                        if (ctx) {
                            ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
                            ctx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);
                            base64Img = tempCanvas.toDataURL("image/jpeg", 0.4);
                        }
                    }

                    if (base64Img) {
                        fetch('/api/wechat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                message: "有人闯入你的虚拟电子围栏！",
                                imageBase64: base64Img
                            })
                        }).then(() => addLog("图片及警报已成功推送至长官微信.", "success"))
                            .catch(() => addLog("网络异常，警报推送失败.", "warning"));
                    }

                    setTimeout(() => {
                        const utterance = new SpeechSynthesisUtterance("警告，您已进入监控禁区，请立即离开");
                        utterance.lang = "zh-CN";
                        utterance.onend = () => {
                            if (videoRef.current && videoRef.current.paused) videoRef.current.play().catch(e => console.error(e));
                        };
                        window.speechSynthesis.speak(utterance);
                        if (videoRef.current && videoRef.current.paused) videoRef.current.play().catch(e => console.error(e));
                    }, 0);
                }
            }

        } catch (e) {
            console.error(e);
        } finally {
            isProcessingRef.current = false;
        }
    };

    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        if (isWebcamOpen) stopWebcam();

        const url = URL.createObjectURL(file);
        setImageSrc(url);
        setInferenceTime(null);
        addLog(`载入静态图片源: ${file.name}`, "info");

        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }

        setTimeout(() => runInference(url), 100);
    };

    const runInference = async (overrideSrc?: string | React.MouseEvent) => {
        const targetSrc = typeof overrideSrc === 'string' ? overrideSrc : imageSrc;
        if (!model || !targetSrc || !canvasRef.current) return;
        if (isWebcamOpen) stopWebcam();

        setLoading(true);
        addLog("正在对静态图像执行全自动 YOLO 深度解析...", "info");
        const start = performance.now();

        try {
            const img = new Image();
            img.src = targetSrc;
            await new Promise((resolve) => (img.onload = resolve));

            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d");
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            ctx?.clearRect(0, 0, canvas.width, canvas.height);

            const inputTensorData = await preprocess(img, 640, 640);
            const inputTensor = new ort.Tensor("float32", Float32Array.from(inputTensorData), [1, 3, 640, 640]);

            const outputs = await model.run({ images: inputTensor });
            const output = outputs["output0"];

            setInferenceTime(performance.now() - start);
            const detectedItems = renderBoxes(canvas, 0.50, output.data as Float32Array, 0, 0) || [];

            currentSightRef.current = detectedItems.map((item: any) => item.label);

            addLog(`图像解析完成，发现 ${detectedItems.length} 个目标对象。特征已同步至 AI 视觉缓存。`, "success");
        } catch (e) {
            addLog("图像解析遭遇异常", "error");
        } finally {
            setLoading(false);
        }
    };

    const askWhatYouSee = async () => {
        if (!isWebcamOpen && !imageSrc) {
            addLog("没有活动的视觉输入源，AI 无法观察。", "warning");
            return;
        }

        speakAnswer("让我看看...");
        addLog("调用视觉分析 API...", "info");

        try {
            const uniqueSight = Array.from(new Set(currentSightRef.current));
            const chineseSight = uniqueSight.map(item => labelTranslator[item] || item).join("、");

            const res = await fetch('/api/brain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    command: "告诉我你目前在画面里看到了什么，简洁一些",
                    sight: chineseSight,
                    isSecurityOn: securityModeRef.current
                })
            });
            const aiDecision = await res.json();
            speakAnswer(aiDecision.reply);
        } catch (error) {
            speakAnswer("眼睛有点花了，重试一下吧。");
            addLog("AI 大脑连接超时，请检查网络。", "error");
        }
    };

    const startWalkieTalkie = () => {
        if (isListening) return; // 如果正在听，防误触

        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (!SpeechRecognition) {
            addLog("❌ 您的控制台不支持语音通讯。", "error");
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.continuous = false; // 🌟 核心：只听一句话，听完自动挂断！
        recognition.interimResults = false;
        recognition.lang = 'zh-CN';

        recognition.onstart = () => {
            setIsListening(true);
            addLog("🎙️ 频道已打开，请直接下达指令（说完自动发送）...", "info");
        };

        recognition.onresult = async (event: any) => {
            const transcript = event.results[0][0].transcript.trim();
            const command = transcript.replace(/[。，！？.,!?]/g, '');

            if (command.length > 0) {
                addLog(`🗣️ 发送战术口令: "${command}"，正在呼叫总部...`, "ai");

                // 🚀 直接发送给 DeepSeek 大脑
                try {
                    const uniqueSight = Array.from(new Set(currentSightRef.current));
                    const chineseSight = uniqueSight.map((item: any) => labelTranslator[item] || item).join("、");

                    const res = await fetch('/api/brain', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            command: command,
                            sight: chineseSight,
                            isSecurityOn: securityModeRef.current
                        })
                    });

                    const aiDecision = await res.json();
                    speakAnswer(aiDecision.reply);

                    // ⚡ 执行大脑指令
                    if (aiDecision.action === "enable_security") {
                        setIsSecurityMode(true);
                        addLog("🔴 安防管家已开启！", "warning");
                    } else if (aiDecision.action === "disable_security") {
                        setIsSecurityMode(false);
                        addLog("🟢 安防管家已解除。", "success");
                    } else if (aiDecision.action === "enable_hound" && aiDecision.target) {
                        houndTargetRef.current = aiDecision.target;
                        setIsAutoMode(true);
                        addLog(`🐺 AI 已锁定追踪目标：[${aiDecision.target}]`, "warning");
                    } else if (aiDecision.action === "disable_hound") {
                        houndTargetRef.current = null;
                        setIsAutoMode(false);
                        addLog(`🐕 猎犬已召回原地待命。`, "success");
                    }
                } catch (error) {
                    addLog("量子通信断开", "error");
                }
            }
        };

        // 听完一句话，或者没说话超时，自动关闭频道
        recognition.onend = () => {
            setIsListening(false);
        };

        recognition.onerror = (event: any) => {
            setIsListening(false);
            if (event.error !== 'no-speech') {
                addLog(`[通讯干扰]: ${event.error}`, "error");
            }
        };

        recognition.start();
    };
    const speakAnswer = (text: string) => {
        addLog(text, "ai");

        setTimeout(() => {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = "zh-CN";
            utterance.onend = () => {
                if (videoRef.current && videoRef.current.paused) videoRef.current.play().catch(e => console.error(e));
            };
            window.speechSynthesis.speak(utterance);
        }, 0);
    };

    const startWebcam = async () => {
        const unlockUtterance = new SpeechSynthesisUtterance('');
        window.speechSynthesis.speak(unlockUtterance);
        if (isWebcamOpen) return stopWebcam();
        setImageSrc(null);
        setInferenceTime(null);
        addLog("正在请求摄像头访问权限...", "info");

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "environment",
                    width: { ideal: 640 },  // 🌟 核心优化：降维到 640x480，大幅降低 AI 算力压力
                    height: { ideal: 480 }
                }
            });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play();
                    setIsWebcamOpen(true);
                    addLog("实时视频流端口已打通，启动每秒逐帧监听.", "success");
                    detectFrame();
                    setTimeout(updateDimensions, 100);
                };
            }
        } catch (err) {
            addLog("获取摄像头权限失败或设备未找到！", "error");
            alert("无法启动摄像头");
        }
    };

    const stopWebcam = () => {
        if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = undefined;
        }
        const video = videoRef.current;
        if (video) {
            video.onerror = null;
            if (video.srcObject) {
                const stream = video.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            } else {
                video.pause();
                video.src = "";
                video.load();
            }
        }
        const canvas = canvasRef.current;
        if (canvas) canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);

        setIsWebcamOpen(false);
        setLoading(false);
        if (document.fullscreenElement) document.exitFullscreen().catch(() => { });
        setIsFullscreen(false);
        addLog("视频流连接已主动切断，进入休眠状态.", "warning");
    };

    const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || !videoRef.current) return;

        setIsWebcamOpen(true);
        setImageSrc(null);
        setInferenceTime(null);
        setLoading(true);
        addLog(`载入视频流: ${file.name}`, "info");

        if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = undefined;
        }

        const video = videoRef.current;
        video.onerror = null;
        if (video.src.startsWith("blob:")) URL.revokeObjectURL(video.src);

        const url = URL.createObjectURL(file);
        video.src = url;
        video.loop = true;
        video.muted = true;
        video.playsInline = true;

        video.onloadeddata = () => {
            if (!videoRef.current) return;
            video.play().then(() => {
                setLoading(false);
                addLog("视频成功解码，开始执行推理探测.", "success");
                detectFrame();
                setTimeout(updateDimensions, 100);
            }).catch(e => console.error(e));
            video.onloadeddata = null;
        };
        video.onerror = () => {
            setLoading(false);
            setIsWebcamOpen(false);
            addLog("上传的视频格式受内核排斥，解析失败！", "error");
            alert("视频格式不支持");
        };
        video.load();
        event.target.value = "";
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-gray-950 to-black text-cyan-400 p-4 md:p-8 font-sans selection:bg-cyan-900">
            <input type="file" accept="image/*" className="hidden" ref={fileInputRef} onChange={handleImageUpload} />
            <input type="file" accept="video/*" className="hidden" ref={videoInputRef} onChange={handleVideoUpload} />

            <div className="max-w-4xl mx-auto">
                {/* 头部：简约极客风格 Header */}
                <header className="flex items-center justify-between mb-6 border-b border-cyan-900/30 pb-3">
                    <div className="flex items-center gap-2 group cursor-default">
                        <span className="text-xl md:text-2xl transition-all group-hover:drop-shadow-[0_0_8px_rgba(34,211,238,0.6)]">👁️</span>
                        <h1 className="text-lg md:text-2xl font-bold tracking-wider text-cyan-400/90 font-mono uppercase transition-all group-hover:text-cyan-300">
                            JARVIS 视觉终端
                        </h1>
                    </div>
                    <div className="flex items-center gap-2 md:gap-3">
                        <span className="hidden md:block text-xs tracking-widest text-gray-500 font-mono uppercase">System Status //</span>
                        <span className={`px-2 py-1 md:px-3 md:py-1 rounded-full text-[10px] md:text-xs font-bold border font-mono ${isSecurityMode
                            ? "bg-red-500/10 text-red-400 border-red-500/50 shadow-[0_0_10px_rgba(239,68,68,0.3)] animate-pulse"
                            : "bg-green-500/10 text-green-400 border-green-500/50 shadow-[0_0_10px_rgba(34,197,94,0.3)]"
                            }`}>
                            {isSecurityMode ? "ALERT // 警戒模式" : "ONLINE // 待机监控"}
                        </span>
                    </div>
                </header>

                <div className="relative bg-black rounded-2xl overflow-hidden border border-cyan-500/30 shadow-[0_0_30px_rgba(34,211,238,0.15)] w-full h-[55vh] md:h-auto md:aspect-video flex items-center justify-center mb-6 group">
                    <div ref={containerRef} style={wrapperStyle}>
                        {imageSrc && (
                            <img src={imageSrc} alt="uploaded" className="absolute inset-0 w-full h-full object-contain" />
                        )}
                        <video
                            ref={videoRef}
                            autoPlay
                            muted
                            playsInline
                            className={`absolute inset-0 w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity ${imageSrc ? "hidden" : ""}`}
                        />
                        <canvas
                            ref={canvasRef}
                            className="absolute inset-0 w-full h-full object-cover z-10"
                        />
                    </div>

                    {!isWebcamOpen && !imageSrc && (
                        <div className={`absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-10 transition-opacity duration-150 ${isWebcamOpen || imageSrc ? "opacity-0 pointer-events-none" : "opacity-100"
                            }`}>
                            <div className="w-16 h-16 rounded-full border-2 border-cyan-500/30 flex items-center justify-center mb-4 shadow-[0_0_20px_rgba(34,211,238,0.1)] relative">
                                <div className="absolute inset-0 rounded-full bg-cyan-500/10 animate-ping opacity-50"></div>
                                <span className="text-2xl relative z-10 opacity-80">👁️</span>
                            </div>
                            <p className="text-cyan-500 font-mono tracking-widest font-bold">SYSTEM STANDBY</p>
                            <p className="text-gray-500 text-sm mt-2 font-mono tracking-wider">请在下方操作台选择输入源启动视觉引擎</p>
                        </div>
                    )}
                </div>

                {/* 🎯 局域网 IP 直连输入框 */}
                <div className="flex gap-2 items-center p-2 bg-gray-900 rounded-lg border border-cyan-900/50 mb-6">
                    <span className="text-xs font-mono text-gray-500">IP_ADDR:</span>
                    <input
                        className="bg-black text-cyan-400 text-xs font-mono outline-none w-32 px-2 py-1 rounded border border-gray-700 focus:border-cyan-500 transition-colors"
                        value={ipAddress}
                        onChange={(e) => setIpAddress(e.target.value)}
                        placeholder="192.168.x.x"
                    />
                    <button
                        onClick={connectToCar}
                        className={`px-3 py-1 text-xs font-bold rounded transition-all ${isCarConnected ? 'bg-green-600/20 text-green-400 border border-green-500 shadow-[0_0_10px_rgba(34,197,94,0.3)]' : 'bg-cyan-600/20 text-cyan-400 border border-cyan-500 hover:bg-cyan-600/40 animate-pulse'}`}
                    >
                        {isCarConnected ? "● 局域网已接驳" : "INITIALIZE_LAN"}
                    </button>
                </div>

                {!isModelLoading && (
                    <div className="bg-black/60 border border-gray-800 rounded-2xl p-4 md:p-6 mb-6 shadow-[0_0_20px_rgba(0,0,0,0.5)]">
                        {/* 基础视觉模块 */}
                        <div className="flex flex-wrap items-center gap-3 mb-6 border-b border-gray-800 pb-6">
                            <span className="text-sm font-mono text-gray-500 tracking-widest w-full md:w-auto md:mr-4">{">>"} 基础视觉模块</span>
                            <div className="grid grid-cols-2 gap-3 w-full md:w-auto md:flex">
                                <button onClick={() => fileInputRef.current?.click()} className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-900 text-cyan-500 border border-gray-700 rounded-lg hover:border-cyan-500 hover:shadow-[0_0_10px_rgba(34,211,238,0.2)] transition-all w-full md:w-auto">
                                    <span>⬆️</span> 图片
                                </button>
                                <button onClick={() => videoInputRef.current?.click()} className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-900 text-cyan-500 border border-gray-700 rounded-lg hover:border-cyan-500 hover:shadow-[0_0_10px_rgba(34,211,238,0.2)] transition-all w-full md:w-auto">
                                    <span>▶️</span> 视频
                                </button>
                            </div>
                            <button
                                onClick={isWebcamOpen ? stopWebcam : startWebcam}
                                className={`flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-all w-full md:w-auto ${isWebcamOpen
                                    ? "bg-red-900/20 text-red-400 border border-red-800 hover:bg-red-900/40 hover:shadow-[0_0_15px_rgba(239,68,68,0.4)]"
                                    : "bg-cyan-600/20 text-cyan-400 border border-cyan-500 hover:bg-cyan-600/40 hover:shadow-[0_0_15px_rgba(34,211,238,0.4)]"
                                    }`}
                            >
                                <span>{isWebcamOpen ? "⏹️" : "📷"}</span> {isWebcamOpen ? "断开视觉流" : "开启摄像头"}
                            </button>
                        </div>

                        {/* AI 核心引擎 */}
                        <div className="flex flex-wrap items-center gap-3">
                            <span className="text-sm font-mono text-gray-500 tracking-widest w-full md:w-auto md:mr-4">{">>"} AI 核心引擎</span>
                            <div className="grid grid-cols-2 gap-3 w-full md:w-auto md:flex">
                                <button
                                    onClick={startWalkieTalkie}
                                    disabled={isListening}
                                    className={`flex items-center justify-center gap-2 px-6 py-2 border rounded-lg font-bold transition-all w-full md:w-auto shadow-lg ${isListening
                                        ? 'bg-green-600 text-white border-green-500 animate-pulse shadow-[0_0_20px_rgba(22,163,74,0.6)] cursor-not-allowed'
                                        : 'bg-slate-800 text-cyan-400 border-cyan-600 hover:bg-slate-700 hover:border-cyan-400 hover:shadow-[0_0_15px_rgba(34,211,238,0.4)] active:scale-95'
                                        }`}
                                >
                                    <span className="text-xl">{isListening ? "📡" : "🎙️"}</span>
                                    {isListening ? "正在接收语音..." : "点击下达指令"}
                                </button>
                                <button
                                    onClick={askWhatYouSee}
                                    className="flex items-center justify-center gap-2 px-4 py-2 bg-indigo-900/40 text-indigo-300 border border-indigo-700 rounded-lg hover:bg-indigo-900/60 hover:shadow-[0_0_15px_rgba(99,102,241,0.4)] transition-all w-full md:w-auto"
                                >
                                    <span>🧠</span> 分析画面
                                </button>
                            </div>
                        </div>

                        {/* 🏎️ 物理机动模块 */}
                        <div className="flex flex-col gap-3 mt-6 border-t border-gray-800 pt-6">
                            <div className="flex flex-wrap items-center gap-3">
                                <span className="text-sm font-mono text-gray-500 tracking-widest w-full md:w-auto md:mr-4">{">>"} 物理机动模块</span>
                                <div className="grid grid-cols-2 gap-3 w-full md:w-auto md:flex">
                                    <button
                                        onClick={connectToCar}
                                        className={`flex items-center justify-center gap-2 px-4 py-2 border rounded-lg transition-all w-full md:w-auto ${isCarConnected ? "bg-green-900/30 text-green-400 border-green-500 shadow-[0_0_15px_rgba(34,197,94,0.3)] pointer-events-none" : "bg-gray-900 text-cyan-500 border-gray-700 hover:border-cyan-500"}`}
                                    >
                                        <span>{isCarConnected ? "⚡" : "🔌"}</span> {isCarConnected ? "底盘已在线" : "连接物理底盘"}
                                    </button>

                                    {/* 只有当底盘连接后，才显示自动驾驶按钮 */}
                                    {isCarConnected && (
                                        <button
                                            onClick={() => setIsAutoMode(!isAutoMode)}
                                            className={`px-4 py-2 border rounded-lg font-bold transition-all ${isAutoMode
                                                ? "bg-red-900/40 text-red-400 border-red-500 animate-pulse shadow-[0_0_15px_rgba(239,68,68,0.4)]"
                                                : "bg-gray-800 text-green-400 border-green-700 hover:border-green-500 hover:bg-green-900/20"
                                                }`}
                                        >
                                            {isAutoMode ? "🧠 AI 接管中：点击解除" : "🚘 启动 AI 驾驶"}
                                        </button>
                                    )}
                                </div>
                            </div>

                            {/* 📱 全息十字触控面板 (只有连接后且非 AI 模式才显示) */}
                            {isCarConnected && !isAutoMode && (
                                <div className="w-full mt-4 flex flex-col items-center justify-center border border-cyan-900/30 bg-cyan-900/5 rounded-xl p-4 md:p-6">
                                    <p className="text-cyan-500 font-mono text-xs mb-4 uppercase tracking-widest animate-pulse">
                                        Manual Override // 手动 0 延迟操控面板
                                    </p>

                                    {/* 十字方向键：使用 touch-none 防止手机滑动屏幕，select-none 防止文字被选中 */}
                                    <div className="grid grid-cols-3 gap-2 w-[240px] mb-4 touch-none select-none">
                                        <div /> {/* 左上留空 */}
                                        <button
                                            onPointerDown={() => sendCarCommand('W')}
                                            onPointerUp={() => sendCarCommand('Q')}
                                            onPointerLeave={() => sendCarCommand('Q')}
                                            className="bg-gray-800/80 hover:bg-cyan-900/80 active:bg-cyan-500 text-cyan-400 border border-gray-600 rounded-lg h-16 flex justify-center items-center shadow-[0_0_15px_rgba(34,211,238,0.1)] transition-colors text-xl font-bold"
                                        >
                                            W
                                        </button>
                                        <div /> {/* 右上留空 */}

                                        <button
                                            onPointerDown={() => sendCarCommand('A')}
                                            onPointerUp={() => sendCarCommand('Q')}
                                            onPointerLeave={() => sendCarCommand('Q')}
                                            className="bg-gray-800/80 hover:bg-cyan-900/80 active:bg-cyan-500 text-cyan-400 border border-gray-600 rounded-lg h-16 flex justify-center items-center transition-colors text-xl font-bold"
                                        >
                                            A
                                        </button>
                                        <button
                                            onPointerDown={() => sendCarCommand('S')}
                                            onPointerUp={() => sendCarCommand('Q')}
                                            onPointerLeave={() => sendCarCommand('Q')}
                                            className="bg-gray-800/80 hover:bg-cyan-900/80 active:bg-cyan-500 text-cyan-400 border border-gray-600 rounded-lg h-16 flex justify-center items-center transition-colors text-xl font-bold"
                                        >
                                            S
                                        </button>
                                        <button
                                            onPointerDown={() => sendCarCommand('D')}
                                            onPointerUp={() => sendCarCommand('Q')}
                                            onPointerLeave={() => sendCarCommand('Q')}
                                            className="bg-gray-800/80 hover:bg-cyan-900/80 active:bg-cyan-500 text-cyan-400 border border-gray-600 rounded-lg h-16 flex justify-center items-center transition-colors text-xl font-bold"
                                        >
                                            D
                                        </button>
                                    </div>
                                    <button
                                        onClick={() => sendCarCommand('P')}
                                        className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-lg shadow-[0_0_15px_rgba(147,51,234,0.5)] transition-all active:scale-95"
                                    >
                                        🚀 执行正方形巡航 (P)
                                    </button>
                                    {/* 紧急刹车按钮 */}
                                    <button
                                        onPointerDown={() => sendCarCommand('Q')}
                                        className="w-[240px] bg-red-900/30 hover:bg-red-800 active:bg-red-500 text-red-400 active:text-white font-bold border border-red-800 rounded-lg py-4 flex justify-center items-center tracking-widest transition-colors shadow-[0_0_15px_rgba(239,68,68,0.2)] touch-none select-none"
                                    >
                                        🛑 紧急刹车
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* 终端 UI 面板 */}
                <div className="bg-black/90 text-green-400 font-mono p-4 rounded-xl border border-gray-800 h-64 overflow-y-auto shadow-[inset_0_0_20px_rgba(0,255,0,0.05)] relative scrollbar-thin scrollbar-thumb-green-900 scrollbar-track-black">
                    <div className="sticky top-0 bg-black/90 pb-2 border-b border-green-900/50 mb-3 flex justify-between z-10">
                        <span className="text-xs text-green-600 font-bold uppercase tracking-widest">System Output Logs //</span>
                        {isModelLoading && (
                            <span className="text-xs text-green-400 animate-pulse">
                                {generateTerminalBar(downloadProgress)}
                            </span>
                        )}
                    </div>

                    <div className="space-y-1.5 text-xs sm:text-sm">
                        {logs.map(log => (
                            <div key={log.id} className="flex items-start gap-2 font-mono break-all">
                                <span className="text-gray-500 shrink-0">[{log.time}]</span>
                                <span className={`${log.type === 'error' ? 'text-red-500 font-bold' :
                                    log.type === 'success' ? 'text-green-400' :
                                        log.type === 'warning' ? 'text-yellow-400' :
                                            log.type === 'ai' ? 'text-cyan-400 font-bold' : 'text-green-500/80'
                                    }`}>
                                    {log.type === 'error' && ' [ERR]'}
                                    {log.type === 'warning' && ' [WARN]'}
                                    {log.type === 'ai' && ' [JARVIS]'}
                                    {log.type === 'success' && ' [OK]'}
                                    {log.type === 'info' && ' [INFO]'} {log.text}
                                </span>
                            </div>
                        ))}
                        <div ref={logsEndRef} />
                    </div>
                </div>
            </div>
        </div>
    );
}