"use client";

import { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web"; // 确保安装了 onnxruntime-web
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { preprocess, renderBoxes } from "@/lib/utils";
import {
    Camera, StopCircle, Play, Loader2,
    Maximize2, Minimize2,
    Zap, Upload
} from "lucide-react";

// --- 1. 关键修改：使用 CDN 加载 WASM (解决手机加载失败问题) ---
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
// @ts-ignore
ort.env.wasm.numThreads = 1; // 手机上单线程通常更稳定

export default function DemoPage() {
    // --- 状态管理 ---
    const [model, setModel] = useState<ort.InferenceSession | null>(null);
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [inferenceTime, setInferenceTime] = useState<number | null>(null);
    const [isWebcamOpen, setIsWebcamOpen] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);

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
    
    // 🔥 关键修改：添加 modelRef 用于动画循环，防止 "session is not defined"
    const modelRef = useRef<ort.InferenceSession | null>(null);

    // --- 2. 初始化加载模型 ---
    useEffect(() => {
        const initModel = async () => {
            try {
                // 确保 yolo11n.onnx 放在 public/model/ 目录下
                const modelPath = `${window.location.origin}/model/yolo11n.onnx`;
                const session = await ort.InferenceSession.create(modelPath, {
                    executionProviders: ["wasm"],
                    graphOptimizationLevel: "all",
                });
                
                // 同时更新 State (UI用) 和 Ref (动画循环用)
                setModel(session);
                modelRef.current = session; 
                
                console.log("模型加载成功!");
            } catch (e) {
                console.error("模型加载失败:", e);
                // 生产环境建议去掉 alert，用 console 即可
                console.log("请检查 public/model/yolo11n.onnx 是否存在");
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
                    facingMode: "environment", // 优先后置
                    width: { ideal: 640 },     // 降低分辨率以提速
                    height: { ideal: 480 }
                },
                audio: false,
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play();
                    setIsWebcamOpen(true);
                    detectFrame(); // 开始循环
                };
            }
        } catch (err) {
            console.error("摄像头启动失败:", err);
            alert("无法访问摄像头，请确认权限。");
        }
    };

    const stopWebcam = () => {
        if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = undefined;
        }

        if (videoRef.current) {
            const video = videoRef.current;
            if (video.srcObject) {
                const stream = video.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            } else {
                video.pause();
                video.src = "";
            }
        }

        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }

        setIsWebcamOpen(false);
        setLoading(false);
        
        if (document.fullscreenElement) {
            document.exitFullscreen().catch(() => { });
        }
    };

    // --- 检测循环 (核心优化部分) ---
    const detectFrame = async () => {
        // 使用 modelRef.current 而不是 model，防止闭包拿不到值
        if (!videoRef.current || !canvasRef.current || !modelRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (video.readyState === 4 && !video.paused && !video.ended) {
            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }

            const now = Date.now();
            
            // 🔥 关键修改：增加间隔到 150ms (每秒~6帧)
            // 之前的 30ms 会让手机 CPU 瞬间 100% 然后卡死
            const FPS_INTERVAL = 150; 
            
            if (now - lastTimeRef.current >= FPS_INTERVAL && !isProcessingRef.current) {
                isProcessingRef.current = true;
                lastTimeRef.current = now;

                try {
                    // 必须先清空画布，否则框会重叠
                    ctx?.clearRect(0, 0, canvas.width, canvas.height);

                    const start = performance.now();

                    const inputTensorData = await preprocess(video, 640, 640);
                    const inputTensor = new ort.Tensor("float32", Float32Array.from(inputTensorData), [1, 3, 640, 640]);
                    
                    // 使用 Ref 进行推理
                    const outputs = await modelRef.current.run({ images: inputTensor });
                    const output = outputs["output0"];

                    const end = performance.now();

                    frameCountRef.current++;
                    // 每5次更新一次 UI 上的时间，减少重绘
                    if (frameCountRef.current % 5 === 0) {
                        setInferenceTime(end - start);
                    }

                    renderBoxes(canvas, 0.25, output.data as Float32Array, 0, 0);

                } catch (e) {
                    console.error("推理报错:", e);
                } finally {
                    isProcessingRef.current = false;
                }
            } else {
                 // 如果没到时间，也要请求下一帧，否则循环会断
                 // 但是这里可以不做清空，保留上一帧的框，视觉更稳定
            }
        }
        requestRef.current = requestAnimationFrame(detectFrame);
    };

    const toggleFullscreen = () => {
        if (!document.fullscreenElement) {
            containerRef.current?.requestFullscreen().then(() => setIsFullscreen(true));
        } else {
            document.exitFullscreen().then(() => setIsFullscreen(false));
        }
    };

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
            alert("视频无法加载");
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
                    <div
                        ref={containerRef}
                        className={`relative flex justify-center items-center bg-black overflow-hidden ${
                            isFullscreen 
                            ? "w-screen h-screen fixed inset-0 z-50 rounded-none" 
                            : "w-full aspect-video md:h-[600px] rounded-lg"
                        }`}
                    >
                        <div className="relative inline-flex max-w-full max-h-full items-center justify-center">
                            <video
                                ref={videoRef}
                                className={`block w-full h-auto max-w-full max-h-full ${!isWebcamOpen ? "hidden" : ""}`}
                                muted
                                playsInline // 必须加，否则 iOS 无法内联播放
                            />

                            {imageSrc && !isWebcamOpen && (
                                <img
                                    src={imageSrc}
                                    alt="Preview"
                                    className="block w-full h-auto max-w-full max-h-full object-contain"
                                    onLoad={(e) => {
                                        const img = e.currentTarget;
                                        if (canvasRef.current) {
                                            canvasRef.current.width = img.naturalWidth;
                                            canvasRef.current.height = img.naturalHeight;
                                        }
                                    }}
                                />
                            )}

                            <canvas
                                ref={canvasRef}
                                className="absolute inset-0 w-full h-full pointer-events-none"
                            />
                        </div>

                        {loading && (
                            <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/50 backdrop-blur-sm">
                                <Loader2 className="h-12 w-12 text-white animate-spin mb-4" />
                                <p className="text-white font-medium">Loading AI...</p>
                            </div>
                        )}

                        {isWebcamOpen && (
                            <button
                                onClick={toggleFullscreen}
                                className="absolute top-4 right-4 z-50 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors"
                            >
                                {isFullscreen ? <Minimize2 className="h-6 w-6" /> : <Maximize2 className="h-6 w-6" />}
                            </button>
                        )}

                        {!imageSrc && !isWebcamOpen && !loading && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-2">
                                <Camera className="w-12 h-12 opacity-50" />
                                <p className="text-sm">点击下方按钮开始检测</p>
                            </div>
                        )}
                    </div>

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

            <div className="mt-8 hidden md:grid grid-cols-3 gap-6 text-center">
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