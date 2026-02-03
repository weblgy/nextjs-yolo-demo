import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Code, ScanFace } from "lucide-react";

export default function Home() {
  return (
    <div className="container mx-auto px-4 py-16 md:py-24">
      {/* Hero Section */}
      <section className="text-center space-y-6 max-w-3xl mx-auto mb-20">
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-slate-900">
          探索 <span className="text-blue-600">视觉智能</span> 与 <span className="text-indigo-600">工程落地</span> 的边界
        </h1>
        <p className="text-lg text-slate-600">
          我是李关宇，一名热衷于将 CV 算法部署到边缘端的开发者。
          这里记录了我的目标检测研究与全栈开发实践。
        </p>
        <div className="flex justify-center gap-4">
          <Link href="/demo">
            <Button size="lg" className="gap-2">
              体验 Web 推理 <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
          <Link href="https://github.com/你的用户名" target="_blank">
            <Button size="lg" variant="outline">查看 GitHub</Button>
          </Link>
        </div>
      </section>

      {/* Feature/Projects Section */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ScanFace className="h-6 w-6 text-blue-500"/>
              目标检测 Demo
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-slate-500 mb-4">
              基于 YOLOv8 + ONNX Runtime Web。
              完全运行在浏览器端，利用 WebAssembly 加速，无需上传图片到服务器，保护隐私且低延迟。
            </p>
            <Link href="/demo" className="text-blue-600 hover:underline font-medium">
              去试试 &rarr;
            </Link>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Code className="h-6 w-6 text-indigo-500"/>
              工程化实践
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-slate-500 mb-4">
              记录了 CI/CD 流水线搭建、Java 高并发与 Next.js 服务端渲染的最佳实践。
              展示如何构建健壮的微服务架构。
            </p>
            <Link href="/blog" className="text-indigo-600 hover:underline font-medium">
              阅读博客 &rarr;
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}