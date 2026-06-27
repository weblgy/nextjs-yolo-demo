"use client";
import { useState, useRef, useEffect } from "react";

// --- 类型定义 ---
interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  sources?: string[];
}

interface DocStatus {
  name: string;
  chunks: number;
  totalChars: number;
}

// --- 示例文档 ---
const SAMPLE_DOC = `人工智能在医疗领域的应用

人工智能（AI）正在深刻改变医疗健康行业。从医学影像分析到药物研发，AI 技术正逐步渗透到医疗的各个环节。

1. 医学影像诊断
深度学习模型在医学影像分析中表现出色。卷积神经网络（CNN）可以自动识别 X 光片、CT 扫描和 MRI 图像中的异常区域。研究表明，在某些特定任务上，AI 辅助诊断的准确率已经达到甚至超过了资深放射科医生的水平。例如，在肺结节检测任务中，AI 模型的灵敏度可达 94% 以上。

2. 药物研发加速
传统药物研发周期通常需要 10-15 年，耗资数十亿美元。AI 技术可以将这一过程大幅缩短。通过分子模拟和深度学习，AI 可以在数百万化合物中快速筛选出有潜力的候选药物。2020 年，第一个完全由 AI 设计的药物分子进入临床试验阶段。

3. 个性化医疗
基于患者的基因组数据、生活习惯和病史，AI 系统可以制定个性化的治疗方案。精准医疗不再是概念——AI 可以帮助医生根据肿瘤的基因突变类型选择最有效的靶向药物。

4. 医疗机器人
手术机器人结合 AI 视觉系统，可以辅助外科医生进行更精准的微创手术。康复机器人通过传感器和自适应算法，为患者提供个性化的康复训练方案。

5. 挑战与展望
尽管 AI 在医疗领域前景广阔，但仍面临数据隐私、算法可解释性、临床验证等挑战。未来，联邦学习等隐私计算技术有望在保护患者数据的同时，实现多中心联合建模。

总结而言，AI 不是要取代医生，而是成为医生的"超级助手"——让诊断更准确、治疗更精准、医疗资源更普惠。`;

export default function RAGPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [docText, setDocText] = useState("");
  const [docName, setDocName] = useState("我的文档");
  const [isIndexing, setIsIndexing] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [docStatus, setDocStatus] = useState<DocStatus | null>(null);
  const [activeTab, setActiveTab] = useState<"upload" | "chat">("upload");
  const [toast, setToast] = useState<{ text: string; type: "success" | "error" | "info" } | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 自动滚动
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 加载文档状态
  useEffect(() => {
    fetchDocStatus();
  }, []);

  const showToast = (text: string, type: "success" | "error" | "info" = "info") => {
    setToast({ text, type });
    setTimeout(() => setToast(null), 3000);
  };

  const fetchDocStatus = async () => {
    try {
      const res = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "status" }),
      });
      const data = await res.json();
      if (data.documents?.length > 0) {
        setDocStatus(data.documents[0]);
        setDocName(data.documents[0].name);
        setActiveTab("chat");
      }
    } catch { /* ignore */ }
  };

  // 索引文档
  const handleIndex = async (text?: string) => {
    const content = text || docText;
    if (!content.trim()) {
      showToast("请先输入或上传文档内容", "error");
      return;
    }

    setIsIndexing(true);
    try {
      const res = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "index", text: content, name: docName }),
      });
      const data = await res.json();

      if (data.success) {
        setDocStatus(data.stats);
        showToast(`文档索引完成！共 ${data.stats.chunkCount} 个片段`, "success");
        setActiveTab("chat");
      } else {
        showToast(data.error || "索引失败", "error");
      }
    } catch {
      showToast("网络异常，索引失败", "error");
    } finally {
      setIsIndexing(false);
    }
  };

  // 发送问题
  const handleQuery = async () => {
    if (!input.trim()) return;
    if (!docStatus) {
      showToast("请先索引文档", "error");
      setActiveTab("upload");
      return;
    }

    const question = input.trim();
    setInput("");
    const userMsg: Message = { id: Date.now(), role: "user", text: question };
    setMessages((prev) => [...prev, userMsg]);
    setIsQuerying(true);

    try {
      const res = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "query", question, docName }),
      });
      const data = await res.json();

      const aiMsg: Message = {
        id: Date.now(),
        role: "assistant",
        text: data.answer || "抱歉，无法生成回答。",
        sources: data.sources,
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { id: Date.now(), role: "assistant", text: "网络异常，请重试。" },
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  // 文件上传
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setDocName(file.name.replace(/\.[^/.]+$/, ""));
    const reader = new FileReader();

    reader.onload = (ev) => {
      const content = ev.target?.result as string;
      setDocText(content);
      handleIndex(content);
    };

    reader.onerror = () => showToast("文件读取失败", "error");

    if (file.type === "application/pdf") {
      showToast("PDF 文件请在本地解析后粘贴文本，或使用 TXT/MD 格式", "info");
      return;
    }

    reader.readAsText(file);
    e.target.value = "";
  };

  // 加载示例文档
  const loadSample = () => {
    setDocText(SAMPLE_DOC);
    setDocName("AI在医疗领域的应用");
    handleIndex(SAMPLE_DOC);
  };

  // 清空对话与文档
  const handleClear = async () => {
    try {
      await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "clear", docName }),
      });
      setMessages([]);
      setDocStatus(null);
      setDocText("");
      showToast("文档和对话已清空", "info");
      setActiveTab("upload");
    } catch {
      showToast("清空失败", "error");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 to-slate-900 text-slate-200 p-4 md:p-8 font-sans">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <header className="flex items-center justify-between mb-6 border-b border-purple-900/30 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xl md:text-2xl">📚</span>
            <h1 className="text-lg md:text-2xl font-bold tracking-wider text-purple-400/90 font-mono uppercase">
              RAG 知识引擎
            </h1>
          </div>
          <div className="flex items-center gap-3">
            {docStatus && (
              <span className="px-2 py-1 rounded-full text-[10px] md:text-xs font-bold border font-mono bg-purple-500/10 text-purple-400 border-purple-500/50">
                {docStatus.chunks} CHUNKS // {docStatus.name}
              </span>
            )}
            <span className="hidden md:block text-xs tracking-widest text-slate-500 font-mono uppercase">
              Retrieval-Augmented Generation //
            </span>
          </div>
        </header>

        {/* Tab Switch */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab("upload")}
            className={`px-4 py-2 text-sm font-mono rounded-lg border transition-all ${
              activeTab === "upload"
                ? "bg-purple-600/20 text-purple-400 border-purple-500"
                : "bg-slate-800 text-slate-500 border-slate-700 hover:border-purple-700"
            }`}
          >
            📄 文档上传
          </button>
          <button
            onClick={() => setActiveTab("chat")}
            className={`px-4 py-2 text-sm font-mono rounded-lg border transition-all ${
              activeTab === "chat"
                ? "bg-purple-600/20 text-purple-400 border-purple-500"
                : "bg-slate-800 text-slate-500 border-slate-700 hover:border-purple-700"
            }`}
          >
            💬 知识问答 {docStatus ? `(${messages.length})` : ""}
          </button>
        </div>

        {/* Upload Panel */}
        {activeTab === "upload" && (
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 mb-6 shadow-[0_0_20px_rgba(0,0,0,0.5)]">
            <div className="flex items-center gap-4 mb-4">
              <input
                className="bg-slate-800 text-purple-300 text-sm font-mono outline-none flex-1 px-3 py-2 rounded-lg border border-slate-700 focus:border-purple-500 transition-colors"
                value={docName}
                onChange={(e) => setDocName(e.target.value)}
                placeholder="文档名称"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2 bg-slate-800 text-purple-400 border border-slate-700 rounded-lg hover:border-purple-500 hover:bg-slate-700 transition-all text-sm font-mono"
              >
                📁 选择文件
              </button>
              <input
                type="file"
                accept=".txt,.md,.csv,.json"
                className="hidden"
                ref={fileInputRef}
                onChange={handleFileUpload}
              />
            </div>

            <textarea
              className="w-full h-64 bg-slate-950 border border-slate-700 rounded-xl p-4 text-purple-300/80 font-mono text-sm outline-none resize-none focus:border-purple-500 transition-colors placeholder:text-slate-600"
              value={docText}
              onChange={(e) => setDocText(e.target.value)}
              placeholder="在此粘贴文档内容，或点击上方按钮选择 TXT/MD 文件...
支持中文、英文、混合文档。系统将自动进行文本分块、分词和索引构建。"
            />

            <div className="flex gap-3 mt-4">
              <button
                onClick={() => handleIndex()}
                disabled={isIndexing || !docText.trim()}
                className="flex items-center gap-2 px-6 py-2.5 bg-purple-600 hover:bg-purple-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-bold rounded-lg transition-all active:scale-95 shadow-[0_0_15px_rgba(147,51,234,0.3)]"
              >
                {isIndexing ? (
                  <>
                    <span className="animate-spin">⏳</span> 索引中...
                  </>
                ) : (
                  <>
                    <span>⚡</span> 索引文档
                  </>
                )}
              </button>
              <button
                onClick={loadSample}
                className="px-4 py-2.5 bg-slate-800 text-slate-400 border border-slate-700 rounded-lg hover:border-purple-500 hover:text-purple-400 transition-all text-sm font-mono"
              >
                📋 加载示例文档
              </button>
              {docStatus && (
                <button
                  onClick={handleClear}
                  className="px-4 py-2.5 bg-red-900/20 text-red-400 border border-red-800 rounded-lg hover:bg-red-900/40 transition-all text-sm font-mono"
                >
                  🗑️ 清空
                </button>
              )}
            </div>

            {docStatus && (
              <div className="mt-4 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <p className="text-purple-400 font-mono text-xs">
                  ✅ 当前文档：<span className="text-purple-300">{docStatus.name}</span>
                  {" "}| {docStatus.chunks} 个片段 | {docStatus.totalChars.toLocaleString()} 字符
                </p>
              </div>
            )}
          </div>
        )}

        {/* Chat Panel */}
        {activeTab === "chat" && (
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-[0_0_20px_rgba(0,0,0,0.5)]">
            {/* Chat Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-800 bg-slate-900/80">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                <span className="text-purple-400 font-mono text-sm">
                  {docStatus ? `📄 ${docStatus.name} (${docStatus.chunks} chunks)` : "未加载文档"}
                </span>
              </div>
              <button
                onClick={handleClear}
                className="text-xs text-slate-600 hover:text-red-400 font-mono transition-colors"
              >
                [CLEAR]
              </button>
            </div>

            {/* Messages */}
            <div className="h-[50vh] overflow-y-auto p-4 space-y-4">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-slate-600">
                  <span className="text-4xl mb-4">📚</span>
                  <p className="font-mono text-sm">基于文档内容提问，AI 将从索引中检索相关内容并生成回答</p>
                  <p className="font-mono text-xs mt-2 text-slate-700">
                    试试："AI在医学影像诊断中的表现如何？"
                  </p>
                </div>
              )}

              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl p-4 ${
                      msg.role === "user"
                        ? "bg-purple-600/20 border border-purple-500/30 text-purple-200"
                        : "bg-slate-800/80 border border-slate-700 text-slate-300"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xs font-mono text-slate-500">
                        {msg.role === "user" ? "YOU" : "AI"}
                      </span>
                      {msg.role === "user" ? (
                        <span className="text-xs">💬</span>
                      ) : (
                        <span className="text-xs">🧠</span>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>

                    {/* Source Citations */}
                    {msg.sources && msg.sources.length > 0 && (
                      <details className="mt-3">
                        <summary className="text-xs font-mono text-purple-500 cursor-pointer hover:text-purple-400">
                          📎 参考来源 ({msg.sources.length})
                        </summary>
                        <div className="mt-2 space-y-1.5 max-h-32 overflow-y-auto">
                          {msg.sources.map((src, i) => (
                            <div
                              key={i}
                              className="text-xs font-mono text-slate-500 bg-slate-900/50 p-2 rounded border border-slate-800"
                            >
                              <span className="text-purple-600">[{i + 1}]</span> {src}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              ))}

              {isQuerying && (
                <div className="flex justify-start">
                  <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-4">
                    <div className="flex items-center gap-2">
                      <span className="animate-pulse text-purple-400">⏳</span>
                      <span className="text-sm text-slate-500 font-mono">检索中...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>

            {/* Input Bar */}
            <div className="p-4 border-t border-slate-800 bg-slate-900/80">
              <div className="flex gap-3">
                <input
                  className="flex-1 bg-slate-800 text-purple-300 text-sm font-mono outline-none px-4 py-3 rounded-xl border border-slate-700 focus:border-purple-500 transition-colors placeholder:text-slate-600"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleQuery()}
                  placeholder={docStatus ? "基于文档提问..." : "请先上传文档"}
                  disabled={isQuerying || !docStatus}
                />
                <button
                  onClick={handleQuery}
                  disabled={isQuerying || !input.trim() || !docStatus}
                  className="px-6 py-3 bg-purple-600 hover:bg-purple-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-bold rounded-xl transition-all active:scale-95"
                >
                  {isQuerying ? "..." : "发送"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Toast */}
      {toast && (
        <div
          className={`fixed bottom-6 left-1/2 -translate-x-1/2 px-6 py-3 rounded-xl border text-sm font-mono z-50 animate-bounce ${
            toast.type === "success"
              ? "bg-green-900/80 border-green-700 text-green-400"
              : toast.type === "error"
              ? "bg-red-900/80 border-red-700 text-red-400"
              : "bg-slate-900/80 border-slate-700 text-purple-400"
          }`}
        >
          {toast.text}
        </div>
      )}
    </div>
  );
}
