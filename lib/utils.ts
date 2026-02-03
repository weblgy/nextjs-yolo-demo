import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const LABELS = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
  "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
  "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
  "toothbrush"
];

/**
 * 预处理：保持比例缩放，并填充灰色背景
 */
export async function preprocess(image: HTMLImageElement | HTMLVideoElement, modelWidth: number, modelHeight: number) {
  const canvas = document.createElement("canvas");
  canvas.width = modelWidth;
  canvas.height = modelHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get canvas context");

  const imgW = "videoWidth" in image ? image.videoWidth : image.width;
  const imgH = "videoHeight" in image ? image.videoHeight : image.height;

  const scale = Math.min(modelWidth / imgW, modelHeight / imgH);
  const newWidth = imgW * scale;
  const newHeight = imgH * scale;
  
  const dx = (modelWidth - newWidth) / 2;
  const dy = (modelHeight - newHeight) / 2;

  // 填充背景
  ctx.fillStyle = "#727272";
  ctx.fillRect(0, 0, modelWidth, modelHeight);
  // 绘制缩放后的图片
  ctx.drawImage(image, dx, dy, newWidth, newHeight);

  const { data } = ctx.getImageData(0, 0, modelWidth, modelHeight);

  const red: number[] = [], green: number[] = [], blue: number[] = [];
  for (let i = 0; i < data.length; i += 4) {
    red.push(data[i] / 255.0);
    green.push(data[i + 1] / 255.0);
    blue.push(data[i + 2] / 255.0);
  }

  return [...red, ...green, ...blue];
}

function iou(box1: number[], box2: number[]) {
  const [x1, y1, x2, y2] = box1;
  const [x1b, y1b, x2b, y2b] = box2;
  const xx1 = Math.max(x1, x1b);
  const yy1 = Math.max(y1, y1b);
  const xx2 = Math.min(x2, x2b);
  const yy2 = Math.min(y2, y2b);
  const w = Math.max(0, xx2 - xx1);
  const h = Math.max(0, yy2 - yy1);
  const inter = w * h;
  const area1 = (x2 - x1) * (y2 - y1);
  const area2 = (x2b - x1b) * (y2b - y1b);
  return inter / (area1 + area2 - inter);
}

/**
 * 渲染函数：修复了坐标偏移，增加了动态线宽
 */
export function renderBoxes(
  canvas: HTMLCanvasElement,
  threshold: number, 
  boxes_data: Float32Array, 
  _ignore: number, 
  _ignore2: number
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  
  // 这里的宽高必须是 Canvas 的实际分辨率（即视频的原始分辨率）
  const imgWidth = canvas.width;
  const imgHeight = canvas.height;
  const modelSize = 640; 

  // --- 1. 严格的坐标反算逻辑 ---
  // 这部分必须和 preprocess 完全一致
  const scale = Math.min(modelSize / imgWidth, modelSize / imgHeight);
  const dx = (modelSize - imgWidth * scale) / 2;
  const dy = (modelSize - imgHeight * scale) / 2;

  // --- 2. 视觉样式优化 ---
  const dynamicLineWidth = Math.max(Math.min(imgWidth / 150, 8), 2); // 稍微调细一点，更精致
  const dynamicFontSize = Math.max(Math.min(imgWidth / 50, 24), 14);
  
  ctx.font = `bold ${dynamicFontSize}px Arial`;
  ctx.lineWidth = dynamicLineWidth;

  const num_classes = 80;
  const num_anchors = 8400; 
  const channel_stride = num_anchors; 
  const candidate_boxes = []; 

  for (let i = 0; i < num_anchors; i++) {
    let maxScore = -Infinity;
    let maxClass = -1;
    for (let c = 0; c < num_classes; c++) {
      const score = boxes_data[(c + 4) * channel_stride + i];
      if (score > maxScore) {
        maxScore = score;
        maxClass = c;
      }
    }

    if (maxScore > threshold) {
      const cx = boxes_data[0 * channel_stride + i];
      const cy = boxes_data[1 * channel_stride + i];
      const w  = boxes_data[2 * channel_stride + i];
      const h  = boxes_data[3 * channel_stride + i];

      // --- 核心修复：先减去偏移，再除以缩放 ---
      // (cx - dx) 才是还原到“无黑边”的缩放图上的坐标
      // 然后除以 scale 还原到原图尺寸
      const x = (cx - dx) / scale;
      const y = (cy - dy) / scale;
      const width = w / scale;
      const height = h / scale;

      const x1 = x - width / 2;
      const y1 = y - height / 2;
      const x2 = x + width / 2;
      const y2 = y + height / 2;

      candidate_boxes.push([x1, y1, x2, y2, maxScore, maxClass]);
    }
  }

  // NMS 处理
  candidate_boxes.sort((a, b) => b[4] - a[4]);
  const result_boxes = [];
  while (candidate_boxes.length > 0) {
    const best = candidate_boxes.shift()!;
    result_boxes.push(best); 
    for (let i = candidate_boxes.length - 1; i >= 0; i--) {
      if (iou(best, candidate_boxes[i]) > 0.45) candidate_boxes.splice(i, 1);
    }
  }

  // 绘制
  result_boxes.forEach((box) => {
    const [x1, y1, x2, y2, score, labelIdx] = box;
    const label = LABELS[labelIdx] || "unknown";
    const text = `${label} ${(score * 100).toFixed(0)}%`;
    
    // 使用鲜艳的颜色
    const color = "#00FF00"; 
    
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const textMetrics = ctx.measureText(text);
    const textWidth = textMetrics.width + 6;
    const textHeight = dynamicFontSize + 6;

    let labelY = y1 - textHeight;
    if (labelY < 0) labelY = y1; 

    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, textWidth, textHeight);

    ctx.fillStyle = "#000000"; // 黑字绿底，对比度高
    ctx.textBaseline = "top";
    ctx.fillText(text, x1 + 3, labelY + 3);
  });
}