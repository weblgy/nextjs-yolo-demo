"use client";

import { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { preprocess, renderBoxes } from "@/lib/utils";
import {
    Camera, StopCircle, Play, ImageIcon, Loader2,
    Maximize2, Minimize2, // <--- 新增这两个图标
    Zap,
    Upload
} from "lucide-react";

// --- 1. 全局配置 ONNX Runtime ---
// 指定 WASM 文件位于 public 根目录
ort.env.wasm.wasmPaths = "/";
// 禁用多线程，防止开发环境出现 SharedArrayBuffer 错误
// @ts-ignore
ort.env.wasm.numThreads = 1;

export default function DemoPage() {
    // --- 状态管理 ---
    const [model, setModel] = useState<ort.InferenceSession | null>(null); // 模型 Session
    const [imageSrc, setImageSrc] = useState<string | null>(null);         // 静态图片路径
    const [loading, setLoading] = useState(false);                         // 推理加载状态
    const [inferenceTime, setInferenceTime] = useState<number | null>(null); // 推理耗时
    const [isWebcamOpen, setIsWebcamOpen] = useState(false);               // 摄像头开关状态

    // --- Refs ---
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const requestRef = useRef<number>(0); // 用于取消 requestAnimationFrame
    const fileInputRef = useRef<HTMLInputElement>(null);
    const videoInputRef = useRef<HTMLInputElement>(null);
    // 在组件内部增加一个 Ref 用来做“锁”，防止推理任务堆积
    const isProcessingRef = useRef(false);
    const lastTimeRef = useRef(0);
    const frameCountRef = useRef(0); // 用来降低 UI 刷新频率
    const containerRef = useRef<HTMLDivElement>(null); // <--- 新增这个 ref
    const [isFullscreen, setIsFullscreen] = useState(false); // <--- 记录全屏状态

    // --- 2. 初始化加载模型 ---
    useEffect(() => {
        const initModel = async () => {
            try {
                // ⚠️ 确保你的 public/model/ 文件夹下有这个文件
                // 如果你用的是 yolov8n，请改成 "yolov8n.onnx"
                const modelPath = "/model/yolo11s.onnx";

                const session = await ort.InferenceSession.create(modelPath, {
                    executionProviders: ["wasm"],
                });
                setModel(session);
                console.log("模型加载成功!");
            } catch (e) {
                console.error("模型加载失败:", e);
                alert("模型加载失败，请检查控制台报错 (通常是路径或文件缺失)");
            }
        };
        initModel();
    }, []);

    // --- 3. 静态图片推理逻辑 (已修复图片消失问题) ---
    const runInference = async () => {
        if (!model || !imageSrc || !canvasRef.current) return;
        if (isWebcamOpen) stopWebcam();

        setLoading(true);
        const start = performance.now();

        try {
            const img = new Image();
            img.src = imageSrc;
            await new Promise((resolve) => (img.onload = resolve));

            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d");

            // 1. 同步尺寸
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;

            // ⚠️ 删除这两行：不要在 Canvas 上画原图，只画框！
            // if (ctx) {
            //     ctx.drawImage(img, 0, 0); 
            // }

            // 确保清空之前的框
            ctx?.clearRect(0, 0, canvas.width, canvas.height);

            // 2. 预处理 & 推理
            const inputTensorData = await preprocess(img, 640, 640);
            const inputTensor = new ort.Tensor("float32", Float32Array.from(inputTensorData), [1, 3, 640, 640]);

            const outputs = await model.run({ images: inputTensor });
            const output = outputs["output0"];

            const end = performance.now();
            setInferenceTime(end - start);

            // 3. 绘制结果
            renderBoxes(canvas, 0.25, output.data as Float32Array, 0, 0);

        } catch (e) {
            console.error(e);
            alert("推理出错，请检查控制台");
        } finally {
            setLoading(false);
        }
    };

    // --- 4. 摄像头处理逻辑 ---

    const startWebcam = async () => {
        // 如果已经开启，点击按钮则关闭
        if (isWebcamOpen) {
            stopWebcam();
            return;
        }

        // 🔥 核心修复：开启摄像头前，清空静态图片
        setImageSrc(null);
        setInferenceTime(null);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    // facingMode: "environment", // 优先使用后置摄像头
                    width: { ideal: 640 },
                    height: { ideal: 480 }     // 适当降低分辨率以提高 FPS
                },
                audio: false,
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play();
                    setIsWebcamOpen(true);
                    detectFrame(); // 开始循环检测
                };
            }
        } catch (err) {
            console.error("摄像头启动失败:", err);
            alert("无法访问摄像头，请检查权限。");
        }
    };

    const stopWebcam = () => {
        // 1. 停止 AI 循环
        if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = undefined;
        }

        if (videoRef.current) {
            const video = videoRef.current;

            // 🔥 核心修复：先解除监听，防止清空 src 时触发报错弹窗 🔥
            video.onerror = null;
            video.onloadeddata = null;

            // 2. 停止播放并清空
            if (video.srcObject) {
                // 如果是摄像头
                const stream = video.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            } else {
                // 如果是视频文件
                video.pause();
                video.src = "";
                video.load();
            }
        }

        // 3. 清理画布
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            // 清空画布，防止绿框残留
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }

        // 4. 重置状态
        setIsWebcamOpen(false);
        setLoading(false);

        // 5. 退出全屏
        if (document.fullscreenElement) {
            document.exitFullscreen().catch(() => { });
        }
    };

    // --- 修改后的检测循环 ---
    const detectFrame = async () => {
        if (!videoRef.current || !canvasRef.current || !model) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (video.readyState === 4 && !video.paused && !video.ended) {
            // 1. 同步尺寸
            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }

            // --- ⚡️ 核心优化：主动节流 (Throttling) ---
            const now = Date.now();
            // 限制至少间隔 80ms (约 12 FPS)，这是 Web 端实时检测的黄金平衡点
            if (now - lastTimeRef.current >= 80 && !isProcessingRef.current) {

                isProcessingRef.current = true;
                lastTimeRef.current = now; // 更新时间戳

                try {
                    const start = performance.now();

                    // 预处理 & 推理
                    const inputTensorData = await preprocess(video, 640, 640);
                    const inputTensor = new ort.Tensor("float32", Float32Array.from(inputTensorData), [1, 3, 640, 640]);
                    const outputs = await model.run({ images: inputTensor });
                    const output = outputs["output0"];

                    const end = performance.now();

                    // --- 优化 UI 更新频率 ---
                    // 只有每 5 帧才更新一次耗时显示，减少 React 重绘压力
                    frameCountRef.current++;
                    if (frameCountRef.current % 5 === 0) {
                        setInferenceTime(end - start);
                    }

                    // 2. 绘制前清空画布 (清除上一帧的框)
                    ctx?.clearRect(0, 0, canvas.width, canvas.height);

                    // 3. 绘制新框 (使用修复后的 utils)
                    renderBoxes(canvas, 0.25, output.data as Float32Array, 0, 0);

                } catch (e) {
                    console.error("推理报错:", e);
                } finally {
                    isProcessingRef.current = false;
                }
            }
        }

        // 依然保持全速循环，但 AI 只有在满足时间间隔时才运行
        requestRef.current = requestAnimationFrame(detectFrame);
    };
    const toggleFullscreen = () => {
        if (!document.fullscreenElement) {
            // 进入全屏
            containerRef.current?.requestFullscreen().then(() => {
                setIsFullscreen(true);
            });
        } else {
            // 退出全屏
            document.exitFullscreen().then(() => {
                setIsFullscreen(false);
            });
        }
    };

    // 监听全屏变化（防止用户按 Esc 退出时状态没更新）
    useEffect(() => {
        const handleChange = () => {
            setIsFullscreen(!!document.fullscreenElement);
        };
        document.addEventListener("fullscreenchange", handleChange);
        return () => document.removeEventListener("fullscreenchange", handleChange);
    }, []);

    // 组件卸载时清理
    useEffect(() => {
        return () => stopWebcam();
    }, []);

    // --- 5. 处理文件上传 ---
    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            // 停止摄像头
            if (isWebcamOpen) stopWebcam();

            const reader = new FileReader();
            reader.onload = (event) => {
                setImageSrc(event.target?.result as string);
                setInferenceTime(null);
                // 清空 Canvas
                if (canvasRef.current) {
                    const ctx = canvasRef.current.getContext("2d");
                    ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                }
            };
            reader.readAsDataURL(file);
        }
    };
    const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file && videoRef.current) {
            setImageSrc(null);
            setInferenceTime(null); // 同时清空之前的推理数据
            // 1. 设置加载状态 (让用户知道我们在处理)
            setLoading(true);
            // 🔥 核心修复：有新视频进来，先销毁旧图片


            // 2. 关闭之前的资源
            if (isWebcamOpen) stopWebcam();
            // 如果之前有 ObjectURL，最好释放掉（可选优化）
            if (videoRef.current.src.startsWith("blob:")) {
                URL.revokeObjectURL(videoRef.current.src);
            }

            // 3. 使用 createObjectURL (这是瞬间完成的，不需要读取文件内容)
            const url = URL.createObjectURL(file);
            videoRef.current.src = url;
            videoRef.current.srcObject = null;

            videoRef.current.loop = true;
            videoRef.current.muted = true;

            // 4. 监听 "canplay" 事件 (表示视频已经缓冲好，可以开始播放了)
            videoRef.current.oncanplay = () => {
                // 只有当视频真的可以播了，才开始
                videoRef.current?.play();
                setIsWebcamOpen(true);
                setLoading(false); // 关闭 Loading
                detectFrame();     // 启动 AI

                // 清除监听器，防止循环触发
                if (videoRef.current) videoRef.current.oncanplay = null;
            };

            // 5. 错误处理
            videoRef.current.onerror = () => {
                setLoading(false);
                alert("视频格式不支持或无法加载");
            };
        }
    };
    // --- 当 imageSrc 改变时，仅调整 Canvas 尺寸，不绘制图片 ---
    useEffect(() => {
        if (imageSrc && canvasRef.current) {
            const img = new Image();
            img.src = imageSrc;
            img.onload = () => {
                const canvas = canvasRef.current!;
                canvas.width = img.width;
                canvas.height = img.height;

                // ⚠️ 删除绘制代码，防止覆盖 img 标签
                const ctx = canvas.getContext("2d");
                if (ctx) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            };
        }
    }, [imageSrc]);

    return (
        <div className="container mx-auto p-4 max-w-6xl">
            <div className="flex flex-col items-center mb-8 space-y-2">
                <h1 className="text-4xl font-extrabold tracking-tight lg:text-5xl bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    YOLO11 Object Detection
                </h1>
                <p className="text-muted-foreground">
                    Next.js + ONNX Runtime Web + WebAssembly
                </p>
            </div>

            <Card className="overflow-hidden border-2 border-slate-100 shadow-xl">
                <CardContent className="p-0">
                    {/* 视觉展示区域 */}
                    {/* 1. 外层容器：使用 Flex 居中，背景黑色 */}
                    {/* 注意：去掉了 min-h-[400px]，改用 fit-content 的逻辑 */}
                    <div
                        ref={containerRef}
                        className={`relative flex justify-center items-center bg-black overflow-hidden rounded-lg border border-slate-800 ${
                            // 全屏时撑满，非全屏时自适应
                            isFullscreen ? "w-screen h-screen" : "w-full h-auto min-h-[480px]"
                            }`}
                    >

                        {/* 2. 内层包装器：三明治结构，内容撑开宽高 */}
                        <div className="relative inline-flex max-w-full max-h-full items-center justify-center">

                            {/* A. 视频层：仅在 isWebcamOpen 为 true 时显示 */}
                            <video
                                ref={videoRef}
                                className={`block w-auto h-auto max-w-full ${isFullscreen ? "max-h-screen" : "max-h-[80vh]"
                                    } ${!isWebcamOpen ? "hidden" : ""}`} // 关键：用 CSS 隐藏而不是销毁 DOM
                                muted
                                playsInline
                            />

                            {/* B. 图片层：仅在有图片且摄像头关闭时显示 */}
                            {/* 修复核心：让 img 标签真实存在，由它决定宽高比 */}
                            {imageSrc && !isWebcamOpen && (
                                <img
                                    src={imageSrc}
                                    alt="Preview"
                                    className={`block w-auto h-auto max-w-full ${isFullscreen ? "max-h-screen" : "max-h-[80vh]"
                                        } object-contain`}
                                    // 图片加载完成后，同步 Canvas 尺寸
                                    onLoad={(e) => {
                                        const img = e.currentTarget;
                                        if (canvasRef.current) {
                                            canvasRef.current.width = img.naturalWidth;
                                            canvasRef.current.height = img.naturalHeight;
                                        }
                                    }}
                                />
                            )}

                            {/* C. Canvas 层：绝对定位覆盖，背景透明 */}
                            <canvas
                                ref={canvasRef}
                                className="absolute inset-0 w-full h-full pointer-events-none"
                            />
                        </div>

                        {/* Loading 遮罩 (代码保持不变，放在最外层 div 里即可) */}
                        {loading && (
                            <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/50 backdrop-blur-sm">
                                <Loader2 className="h-12 w-12 text-white animate-spin mb-4" />
                                <p className="text-white font-medium">正在初始化 AI...</p>
                            </div>
                        )}

                        {/* 全屏按钮 (代码保持不变) */}
                        {isWebcamOpen && (
                            <button
                                onClick={toggleFullscreen}
                                className="absolute top-4 right-4 z-50 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors"
                            >
                                {isFullscreen ? <Minimize2 className="h-6 w-6" /> : <Maximize2 className="h-6 w-6" />}
                            </button>
                        )}

                        {/* 空状态提示 (代码保持不变) */}
                        {!imageSrc && !isWebcamOpen && !loading && (
                            /* ... 你的空状态代码 ... */
                            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                                <p>请上传视频或开启摄像头</p>
                            </div>
                        )}
                    </div>

                    {/* 控制栏 */}
                    <div className="p-6 bg-white border-t flex flex-col sm:flex-row gap-4 justify-between items-center">

                        {/* 状态显示 */}
                        <div className="flex items-center gap-4 text-sm font-medium">
                            <div className={`flex items-center gap-2 ${model ? 'text-green-600' : 'text-orange-500'}`}>
                                <div className={`w-3 h-3 rounded-full ${model ? 'bg-green-500' : 'bg-orange-400 animate-pulse'}`} />
                                {model ? "模型已加载" : "加载模型中..."}
                            </div>
                            {inferenceTime && (
                                <div className="flex items-center gap-2 text-blue-600">
                                    <Zap className="h-4 w-4 fill-current" />
                                    {inferenceTime.toFixed(1)} ms
                                </div>
                            )}
                        </div>

                        {/* 按钮组 */}
                        <div className="flex gap-3">
                            <input
                                type="file"
                                id="upload"
                                className="hidden"
                                accept="image/*"
                                ref={fileInputRef}
                                onChange={handleImageUpload}
                            />

                            <Button
                                variant="outline"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isWebcamOpen}
                            >
                                <Upload className="mr-2 h-4 w-4" />
                                {imageSrc ? "换一张" : "上传图片"}
                            </Button>

                            {/* 隐藏的视频 input */}
                            <input
                                type="file"
                                id="upload-video"
                                className="hidden"
                                accept="video/*"
                                ref={videoInputRef}
                                onChange={handleVideoUpload}
                            />

                            {/* 上传视频按钮 */}
                            <Button
                                variant="outline"
                                onClick={() => videoInputRef.current?.click()}
                                disabled={isWebcamOpen}
                            >
                                <Play className="mr-2 h-4 w-4" />
                                上传视频检测
                            </Button>

                            <Button
                                onClick={runInference}
                                disabled={!model || !imageSrc || loading || isWebcamOpen}
                                className="min-w-[120px]"
                            >
                                {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                                静态检测
                            </Button>

                            <Button
                                variant={isWebcamOpen ? "destructive" : "default"}
                                onClick={startWebcam} // 这里不用变，startWebcam 里有 stop 逻辑
                                disabled={!model}
                                className={isWebcamOpen ? "animate-pulse" : ""}
                            >
                                {isWebcamOpen ? (
                                    <StopCircle className="mr-2 h-4 w-4" />
                                ) : (
                                    <Camera className="mr-2 h-4 w-4" />
                                )}

                                {/* --- 修改这里的文案逻辑 --- */}
                                {isWebcamOpen
                                    ? (videoRef.current?.srcObject ? "关闭摄像头" : "停止视频")
                                    : "开启摄像头"
                                }
                            </Button>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Footer Info */}
            <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
                <div className="p-4 rounded-lg bg-slate-50">
                    <h3 className="font-bold text-slate-800">100% 本地隐私</h3>
                    <p className="text-sm text-slate-500 mt-1">你的图片和视频流完全在浏览器内处理，不会上传到服务器。</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-50">
                    <h3 className="font-bold text-slate-800">YOLO11s 加持</h3>
                    <p className="text-sm text-slate-500 mt-1">使用最新 SOTA 模型，平衡速度与精度。</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-50">
                    <h3 className="font-bold text-slate-800">WebAssembly</h3>
                    <p className="text-sm text-slate-500 mt-1">通过 ONNX Runtime 实现接近原生的推理速度。</p>
                </div>
            </div>
        </div>
    );
}