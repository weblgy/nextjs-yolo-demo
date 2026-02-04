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

// CDN 配置
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
// @ts-ignore
ort.env.wasm.numThreads = 1;

export default function DemoPage() {
    // --- 状态 ---
    const [model, setModel] = useState<ort.InferenceSession | null>(null);
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [inferenceTime, setInferenceTime] = useState<number | null>(null);
    const [isWebcamOpen, setIsWebcamOpen] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    
    // 控制 Video 和 Canvas 容器尺寸的状态
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
    
    // 逻辑 Refs
    const requestRef = useRef<number>(0);
    const smallCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const isProcessingRef = useRef(false);
    const lastTimeRef = useRef(0);
    const frameCountRef = useRef(0);
    const modelRef = useRef<ort.InferenceSession | null>(null);

    // 初始化离屏画布
    useEffect(() => {
        const sc = document.createElement('canvas');
        sc.width = 640;
        sc.height = 640;
        smallCanvasRef.current = sc;
    }, []);

    // 加载模型
    useEffect(() => {
        const initModel = async () => {
            try {
                const modelPath = `${window.location.origin}/model/yolo11n.onnx`;
                const session = await ort.InferenceSession.create(modelPath, {
                    executionProviders: ["wasm"],
                    graphOptimizationLevel: "all",
                });
                setModel(session);
                modelRef.current = session; 
                console.log("模型加载成功");
            } catch (e) {
                console.error("加载失败:", e);
            }
        };
        initModel();
    }, []);

    // --- 核心：尺寸自适应计算 ---
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

    // 监听窗口变化重算尺寸
    useEffect(() => {
        const timer = setTimeout(updateDimensions, 100);
        window.addEventListener('resize', updateDimensions);
        return () => {
            clearTimeout(timer);
            window.removeEventListener('resize', updateDimensions);
        };
    }, [isFullscreen, imageSrc, isWebcamOpen]);

    // --- 核心：检测循环 ---
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

            // Letterbox 预处理：保持比例缩放并居中
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

            renderBoxes(canvas, 0.50, output.data as Float32Array, 0, 0);

        } catch (e) {
            console.error(e);
        } finally {
            isProcessingRef.current = false;
        }
    };

    // --- 静态图片推理 ---
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

            setInferenceTime(performance.now() - start);
            renderBoxes(canvas, 0.50, output.data as Float32Array, 0, 0);

        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    // --- 功能控制 ---
    const startWebcam = async () => {
        if (isWebcamOpen) return stopWebcam();
        setImageSrc(null);
        setInferenceTime(null);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "environment",
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false,
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                // 监听元数据加载，触发一次尺寸计算
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play();
                    setIsWebcamOpen(true);
                    detectFrame();
                    setTimeout(updateDimensions, 100); 
                };
            }
        } catch (err) {
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
        if (document.fullscreenElement) document.exitFullscreen().catch(()=>{});
        setIsFullscreen(false);
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
                setIsWebcamOpen(true);
                setLoading(false);
                detectFrame();
                setTimeout(updateDimensions, 100);
            }).catch(e => console.error(e));
            video.onloadeddata = null;
        };
        video.onerror = () => {
            setLoading(false);
            alert("视频格式不支持");
        };
        video.load();
        event.target.value = "";
    };

    const toggleFullscreen = () => {
        setIsFullscreen(!isFullscreen);
        // 尝试调用原生 API
        if (containerRef.current && !isFullscreen) {
            containerRef.current.requestFullscreen().catch(() => {});
        } else if (document.fullscreenElement) {
            document.exitFullscreen().catch(() => {});
        }
    };

    return (
        <div className="container mx-auto p-2 md:p-4 max-w-6xl">
            <div className="flex flex-col items-center mb-6 space-y-2">
                <h1 className="text-2xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent text-center">
                    YOLO11 Object Detection
                </h1>
                <p className="text-xs md:text-base text-muted-foreground text-center">
                    Next.js + ONNX Runtime Web
                </p>
            </div>

            <Card className="overflow-hidden border-2 border-slate-100 shadow-xl">
                <CardContent className="p-0">
                    <div
                        ref={containerRef}
                        className={`relative flex justify-center items-center bg-black overflow-hidden transition-all duration-300 ${
                            isFullscreen 
                            ? "fixed inset-0 z-[9999] w-screen h-screen rounded-none" 
                            : imageSrc 
                                ? "w-full h-[60vh] rounded-lg"
                                : "w-full aspect-video md:h-[600px] rounded-lg"
                        }`}
                    >
                        {/* 中间层包装器：尺寸严格等于内容显示尺寸 */}
                        <div style={wrapperStyle}>
                            <video
                                ref={videoRef}
                                className={`block w-full h-full object-fill ${!isWebcamOpen ? "hidden" : ""}`}
                                muted
                                playsInline 
                                webkit-playsinline="true"
                            />

                            {imageSrc && !isWebcamOpen && (
                                <img
                                    src={imageSrc}
                                    alt="Preview"
                                    className="block w-full h-full object-fill"
                                    onLoad={(e) => {
                                        const img = e.currentTarget;
                                        if (canvasRef.current) {
                                            canvasRef.current.width = img.naturalWidth;
                                            canvasRef.current.height = img.naturalHeight;
                                        }
                                        updateDimensions(); 
                                    }}
                                />
                            )}

                            <canvas
                                ref={canvasRef}
                                className="absolute inset-0 w-full h-full pointer-events-none z-10"
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
                                className="absolute top-4 right-4 z-50 p-3 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors active:scale-95"
                            >
                                {isFullscreen ? <Minimize2 className="h-6 w-6" /> : <Maximize2 className="h-6 w-6" />}
                            </button>
                        )}

                        {!imageSrc && !isWebcamOpen && !loading && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-2 h-64">
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
                                onChange={(e) => {
                                    const file = e.target.files?.[0];
                                    if(file) {
                                        if(isWebcamOpen) stopWebcam();
                                        const reader = new FileReader();
                                        reader.onload = (ev) => {
                                            setImageSrc(ev.target?.result as string);
                                            setInferenceTime(null);
                                            setTimeout(updateDimensions, 100);
                                        };
                                        reader.readAsDataURL(file);
                                    }
                                    e.target.value = "";
                                }}
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