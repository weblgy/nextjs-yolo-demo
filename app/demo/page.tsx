"use client";

import { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { preprocess, renderBoxes } from "@/lib/utils";
import {
    Camera, StopCircle, Play, Loader2,
    Maximize2, Minimize2,
    Zap, Upload
} from "lucide-react";

// --- 1. 全局配置 ONNX Runtime ---
ort.env.wasm.wasmPaths = "/";
// @ts-ignore
ort.env.wasm.numThreads = 1;

export default function DemoPage() {
    // --- 状态管理 ---
    const [model, setModel] = useState<ort.InferenceSession | null>(null);
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [inferenceTime, setInferenceTime] = useState<number | null>(null);
    const [isWebcamOpen, setIsWebcamOpen] = useState(false);

    // --- Refs ---
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const requestRef = useRef<number>(0);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const videoInputRef = useRef<HTMLInputElement>(null);
    const isProcessingRef = useRef(false);
    const lastTimeRef = useRef(0);
    const frameCountRef = useRef(0);
    const containerRef = useRef<HTMLDivElement>(null);
    const [isFullscreen, setIsFullscreen] = useState(false);


    // --- 2. 初始化加载模型 ---
    useEffect(() => {
        const initModel = async () => {
            try {
                // 建议使用 yolo11n.onnx (nano版本) 以获得移动端最佳速度
                const modelPath = "/model/yolo11n.onnx";
                const session = await ort.InferenceSession.create(modelPath, {
                    executionProviders: ["wasm"],
                });
                setModel(session);
                console.log("模型加载成功!");
            } catch (e) {
                console.error("模型加载失败:", e);
                alert("模型加载失败，请检查 public/model 目录");
            }
        };
        initModel();
    }, []);

    // --- 3. 静态图片推理逻辑 ---
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

            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;

            ctx?.clearRect(0, 0, canvas.width, canvas.height);

            const inputTensorData = await preprocess(img, 640, 640);
            const inputTensor = new ort.Tensor("float32", Float32Array.from(inputTensorData), [1, 3, 640, 640]);

            const outputs = await model.run({ images: inputTensor });
            const output = outputs["output0"];

            const end = performance.now();
            setInferenceTime(end - start);

            renderBoxes(canvas, 0.25, output.data as Float32Array, 0, 0);

        } catch (e) {
            console.error(e);
            alert("推理出错");
        } finally {
            setLoading(false);
        }
    };

    // --- 4. 摄像头处理逻辑 ---
    const startWebcam = async () => {
        if (isWebcamOpen) {
            stopWebcam();
            return;
        }

        setImageSrc(null);
        setInferenceTime(null);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    // 🔥 移动端核心配置：优先后置，限制分辨率以提高性能
                    facingMode: "environment",
                    width: { ideal: 640 }, // 降低分辨率有助于提高 Canvas 绘制速度
                    height: { ideal: 480 }
                },
                audio: false,
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play();
                    setIsWebcamOpen(true);
                    detectFrame();
                };
            }
        } catch (err) {
            console.error("摄像头启动失败:", err);
            alert("无法访问摄像头，请确认已授予权限且在 HTTPS 环境下运行。");
        }
    };

    const stopWebcam = () => {
        if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = undefined;
        }

        if (videoRef.current) {
            const video = videoRef.current;
            video.onerror = null;
            video.onloadeddata = null;

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

        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }

        setIsWebcamOpen(false);
        setLoading(false);

        // 退出全屏
        if (document.fullscreenElement) {
            document.exitFullscreen().catch(() => { });
        }
    };

    // --- 🔥🔥🔥 核心修改：优化的检测循环 ---
    const detectFrame = async () => {
        if (!videoRef.current || !canvasRef.current || !model) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (video.readyState === 4 && !video.paused && !video.ended) {
            // 尺寸同步
            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }

            const now = Date.now();
            // 🔥🔥🔥 降频处理：由 30ms 改为 150ms (大约每秒检测 6 次)
            // 这对手机至关重要，给 CPU 喘息时间，避免界面卡死
            if (now - lastTimeRef.current >= 150 && !isProcessingRef.current) {
                isProcessingRef.current = true;
                lastTimeRef.current = now;

                try {
                    const start = performance.now();

                    const inputTensorData = await preprocess(video, 640, 640);
                    const inputTensor = new ort.Tensor("float32", Float32Array.from(inputTensorData), [1, 3, 640, 640]);
                    const outputs = await model.run({ images: inputTensor });
                    const output = outputs["output0"];

                    const end = performance.now();

                    frameCountRef.current++;
                    // 每5次检测更新一次时间显示，避免 UI 闪烁
                    if (frameCountRef.current % 5 === 0) {
                        setInferenceTime(end - start);
                    }

                    // 清除画布并重绘
                    ctx?.clearRect(0, 0, canvas.width, canvas.height);
                    renderBoxes(canvas, 0.25, output.data as Float32Array, 0, 0);

                } catch (e) {
                    console.error("推理报错:", e);
                } finally {
                    isProcessingRef.current = false;
                }
            }
        }
        requestRef.current = requestAnimationFrame(detectFrame);
    };

    const toggleFullscreen = () => {
        if (!containerRef.current) return;
        
        if (!document.fullscreenElement) {
            containerRef.current.requestFullscreen().catch(err => {
                console.log("全屏被拦截，尝试使用 CSS 伪全屏", err);
                setIsFullscreen(true); // 即使 API 失败，也切换 React 状态来触发 CSS 变化
            });
        } else {
            document.exitFullscreen().catch(() => {});
        }
    };

    // 监听全屏变化事件（处理 ESC 键退出等情况）
    useEffect(() => {
        const handleChange = () => setIsFullscreen(!!document.fullscreenElement);
        document.addEventListener("fullscreenchange", handleChange);
        return () => document.removeEventListener("fullscreenchange", handleChange);
    }, []);

    useEffect(() => {
        return () => stopWebcam();
    }, []);

    // --- 文件上传处理 ---
    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            if (isWebcamOpen) stopWebcam();
            const reader = new FileReader();
            reader.onload = (event) => {
                setImageSrc(event.target?.result as string);
                setInferenceTime(null);
                if (canvasRef.current) {
                    const ctx = canvasRef.current.getContext("2d");
                    ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                }
            };
            reader.readAsDataURL(file);
        }
        e.target.value = "";
    };

    const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || !videoRef.current) return;

        setImageSrc(null);
        setInferenceTime(null);
        setLoading(true);

        if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = undefined;
        }

        if (videoRef.current.src.startsWith("blob:")) {
            URL.revokeObjectURL(videoRef.current.src);
        }

        const url = URL.createObjectURL(file);
        videoRef.current.src = url;
        videoRef.current.srcObject = null;
        videoRef.current.loop = true;
        videoRef.current.muted = true;

        videoRef.current.oncanplay = () => {
            if (!videoRef.current) return;
            videoRef.current.play();
            setIsWebcamOpen(true);
            setLoading(false);
            detectFrame();
            videoRef.current.oncanplay = null;
        };

        videoRef.current.onerror = () => {
            setLoading(false);
            alert("视频无法加载或格式不支持");
        };

        event.target.value = "";
    };

    // imageSrc 改变时重置 Canvas
    useEffect(() => {
        if (imageSrc && canvasRef.current) {
            const img = new Image();
            img.src = imageSrc;
            img.onload = () => {
                const canvas = canvasRef.current!;
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext("2d");
                ctx?.clearRect(0, 0, canvas.width, canvas.height);
            };
        }
    }, [imageSrc]);

    return (
        <div className="container mx-auto p-2 md:p-4 max-w-6xl">
            <div className="flex flex-col items-center mb-6 space-y-2">
                <h1 className="text-2xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent text-center">
                    YOLO11 Object Detection
                </h1>
                <p className="text-xs md:text-base text-muted-foreground text-center">
                    Next.js + ONNX Runtime Web + WebAssembly
                </p>
            </div>

            <Card className="overflow-hidden border-2 border-slate-100 shadow-xl">
                <CardContent className="p-0">
              {/* --- 视觉展示区域 (PC/手机 完美适配版) --- */}
                    <div
                        ref={containerRef}
                        className={`relative flex justify-center items-center bg-black overflow-hidden transition-all duration-300 ${
                            isFullscreen 
                                ? "fixed inset-0 z-50 w-screen h-screen" // 全屏：占满屏幕
                                : "w-full min-h-[300px] rounded-lg"      // 普通：PC上由内容撑开，给个最小高度防止塌陷
                        }`}
                    >
                        {/* 包装器核心修改：
                            1. relative: 作为 Canvas 的定位基准
                            2. w-auto / h-auto: 让它紧贴视频的大小，这样 Canvas 才能精准覆盖
                        */}
                        <div className={`relative flex items-center justify-center ${isFullscreen ? "w-full h-full" : "w-auto h-auto"}`}>

                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                webkit-playsinline="true"
                                muted
                                // 🔥🔥🔥 核心修改在这里 🔥🔥🔥
                                // 1. 手机 (默认): w-full (占满宽), h-auto (高自适应)
                                // 2. PC (md:): w-auto (宽自适应), h-[600px] (限制高度，防止太巨大)
                                className={`block ${
                                    isFullscreen 
                                        ? "w-full h-full object-contain" 
                                        : "w-full h-auto md:w-auto md:max-h-[600px] md:max-w-full object-contain"
                                } ${!isWebcamOpen ? "hidden" : ""}`}
                            />

                            {imageSrc && !isWebcamOpen && (
                                <img
                                    src={imageSrc}
                                    alt="Preview"
                                    // 同上，保持图片在 PC 上不要太大
                                    className={`block ${
                                        isFullscreen 
                                            ? "w-full h-full object-contain" 
                                            : "w-full h-auto md:w-auto md:max-h-[600px] md:max-w-full object-contain"
                                    }`}
                                />
                            )}

                            {/* Canvas 画布 - 永远覆盖在上面的元素上 */}
                            <canvas
                                ref={canvasRef}
                                className="absolute inset-0 w-full h-full pointer-events-none object-contain"
                            />
                        </div>

                        {/* Loading 状态 */}
                        {loading && (
                            <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/50 backdrop-blur-sm">
                                <Loader2 className="h-12 w-12 text-white animate-spin mb-4" />
                                <p className="text-white font-medium">Loading AI...</p>
                            </div>
                        )}

                        {/* 全屏切换按钮 */}
                        {isWebcamOpen && (
                            <button
                                onClick={toggleFullscreen}
                                className="absolute top-4 right-4 z-[60] p-3 bg-black/40 hover:bg-black/60 backdrop-blur-md text-white rounded-full transition-all border border-white/20"
                            >
                                {isFullscreen ? <Minimize2 className="h-6 w-6" /> : <Maximize2 className="h-6 w-6" />}
                            </button>
                        )}

                        {/* 空状态提示 */}
                        {!imageSrc && !isWebcamOpen && !loading && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-2">
                                <Camera className="w-12 h-12 opacity-50" />
                                <p className="text-sm">点击下方按钮开始检测</p>
                            </div>
                        )}
                    </div>

                    {/* 控制栏 - 保持不变 */}
                    <div className="p-4 bg-white border-t flex flex-col sm:flex-row gap-4 justify-between items-center">
                        <div className="flex items-center gap-4 text-sm font-medium w-full sm:w-auto justify-between sm:justify-start">
                            <div className={`flex items-center gap-2 ${model ? 'text-green-600' : 'text-orange-500'}`}>
                                <div className={`w-2 h-2 md:w-3 md:h-3 rounded-full ${model ? 'bg-green-500' : 'bg-orange-400 animate-pulse'}`} />
                                {model ? "Ready" : "Loading..."}
                            </div>
                            {inferenceTime && (
                                <div className="flex items-center gap-2 text-blue-600">
                                    <Zap className="h-4 w-4 fill-current" />
                                    {inferenceTime.toFixed(0)}ms
                                </div>
                            )}
                        </div>

                        <div className="grid grid-cols-2 sm:flex gap-2 w-full sm:w-auto">
                            <input
                                type="file"
                                className="hidden"
                                accept="image/*"
                                ref={fileInputRef}
                                onChange={handleImageUpload}
                            />
                            <Button
                                variant="outline"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isWebcamOpen}
                                className="w-full sm:w-auto"
                            >
                                <Upload className="mr-2 h-4 w-4" />
                                图片
                            </Button>

                            <input
                                type="file"
                                className="hidden"
                                accept="video/*"
                                ref={videoInputRef}
                                onChange={handleVideoUpload}
                            />
                            <Button
                                variant="outline"
                                onClick={() => videoInputRef.current?.click()}
                                disabled={isWebcamOpen}
                                className="w-full sm:w-auto"
                            >
                                <Play className="mr-2 h-4 w-4" />
                                视频
                            </Button>

                            <Button
                                onClick={runInference}
                                disabled={!model || !imageSrc || loading || isWebcamOpen}
                                className="w-full sm:w-auto col-span-2 sm:col-span-1"
                            >
                                静态检测
                            </Button>

                            <Button
                                variant={isWebcamOpen ? "destructive" : "default"}
                                onClick={startWebcam}
                                disabled={!model}
                                className={`w-full sm:w-auto col-span-2 sm:col-span-1 ${isWebcamOpen ? "animate-pulse" : ""}`}
                            >
                                {isWebcamOpen ? <StopCircle className="mr-2 h-4 w-4" /> : <Camera className="mr-2 h-4 w-4" />}
                                {isWebcamOpen ? "停止" : "摄像头"}
                            </Button>
                        </div>
                    </div>
                </CardContent>
            </Card>
              {/* 底部信息，手机上隐藏或缩小 */}
            <div className="mt-8 hidden md:grid grid-cols-3 gap-6 text-center">
                {/* ... 保持原样 ... */}
                <div className="p-4 rounded-lg bg-slate-50">
                    <h3 className="font-bold text-slate-800">100% 本地隐私</h3>
                    <p className="text-sm text-slate-500 mt-1">你的图片和视频流完全在浏览器内处理。</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-50">
                    <h3 className="font-bold text-slate-800">YOLO11s</h3>
                    <p className="text-sm text-slate-500 mt-1">SOTA 实时目标检测。</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-50">
                    <h3 className="font-bold text-slate-800">WebAssembly</h3>
                    <p className="text-sm text-slate-500 mt-1">ONNX Runtime 原生速度。</p>
                </div>
            </div>
        </div>
    );
}