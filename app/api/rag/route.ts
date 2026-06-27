import { NextResponse } from 'next/server';
import OpenAI from 'openai';

// ============================================================
// 🧠 RAG (Retrieval-Augmented Generation) 引擎
// 实现完整的文档索引 → 语义检索 → 增强生成流水线
// ============================================================

const openai = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: 'https://api.deepseek.com/v1',
});

// --- 类型定义 ---
interface Chunk {
  id: number;
  text: string;
  docName: string;
  tokens: Map<string, number>; // 词频表，用于 TF-IDF 检索
}

// --- 内存文档库 ---
const documentStore = new Map<string, Chunk[]>();

// --- 停用词表（中英文常见停用词）---
const STOP_WORDS = new Set([
  'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
  'in', 'with', 'to', 'for', 'of', 'that', 'this', 'was', 'are',
  'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
  'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall',
  'you', 'your', 'we', 'our', 'they', 'them', 'their', 'its', 'it',
  '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
  '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
  '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
  '所', '被', '从', '而', '以', '之', '于', '及', '与', '或', '但',
]);

// ============================================================
// 🔪 文本分块器 (Recursive Character Text Splitter)
// ============================================================
function chunkText(
  text: string,
  docName: string,
  chunkSize = 500,
  overlap = 50
): Chunk[] {
  const chunks: Chunk[] = [];
  let id = 0;

  // 1. First split by double newlines (paragraphs)
  const paragraphs = text.split(/\n\n+/).filter((p) => p.trim().length > 0);

  for (const paragraph of paragraphs) {
    if (paragraph.length <= chunkSize) {
      chunks.push(makeChunk(paragraph, docName, id++));
      continue;
    }

    // 2. Split long paragraphs by sentences
    const sentences = paragraph.split(/(?<=[。！？.!?\n])\s*/);
    let buffer = '';

    for (const sentence of sentences) {
      if ((buffer + sentence).length > chunkSize && buffer.length > 0) {
        chunks.push(makeChunk(buffer.trim(), docName, id++));
        // Overlap: keep last `overlap` chars
        const keepStart = Math.max(0, buffer.length - overlap);
        buffer = buffer.slice(keepStart) + sentence;
      } else {
        buffer += (buffer ? ' ' : '') + sentence;
      }
    }

    if (buffer.trim().length > 0) {
      chunks.push(makeChunk(buffer.trim(), docName, id++));
    }
  }

  return chunks;
}

function makeChunk(text: string, docName: string, id: number): Chunk {
  return { id, text, docName, tokens: tokenize(text) };
}

// ============================================================
// 📊 分词与词频统计
// ============================================================
function tokenize(text: string): Map<string, number> {
  const freq = new Map<string, number>();

  // 中文按单字+双字组合分词；英文按空格分词
  const tokens: string[] = [];

  // Extract Chinese characters (single + bigram)
  const chineseChars = text.match(/[一-鿿]+/g) || [];
  for (const word of chineseChars) {
    // Bigrams
    for (let i = 0; i < word.length - 1; i++) {
      tokens.push(word.slice(i, i + 2));
    }
    // Single chars
    for (const char of word) {
      tokens.push(char);
    }
  }

  // Extract English/alphabetic tokens
  const englishWords = text.match(/[a-zA-Z]+/g) || [];
  for (const word of englishWords) {
    if (word.length > 1) tokens.push(word.toLowerCase());
  }

  // Count frequencies, filter stop words
  for (const token of tokens) {
    if (STOP_WORDS.has(token) || token.length < 2) continue;
    freq.set(token, (freq.get(token) || 0) + 1);
  }

  return freq;
}

// ============================================================
// 🔍 TF-IDF 检索器
// ============================================================
function searchChunks(query: string, allChunks: Chunk[], topK = 5): Chunk[] {
  const queryTokens = tokenize(query);

  // Compute document frequency (how many chunks contain each token)
  const df = new Map<string, number>();
  for (const chunk of allChunks) {
    for (const token of chunk.tokens.keys()) {
      df.set(token, (df.get(token) || 0) + 1);
    }
  }

  const totalDocs = allChunks.length;

  // Score each chunk
  const scored = allChunks.map((chunk) => {
    let score = 0;
    for (const [qToken, qFreq] of queryTokens) {
      const tf = chunk.tokens.get(qToken) || 0;
      if (tf === 0) continue;
      const idf = Math.log((totalDocs + 1) / ((df.get(qToken) || 0) + 1)) + 1;
      score += tf * idf * qFreq;
    }
    return { chunk, score };
  });

  // Sort by score, return top-K
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).filter((s) => s.score > 0).map((s) => s.chunk);
}

// ============================================================
// 🤖 RAG 增强生成
// ============================================================
async function generateAnswer(
  question: string,
  context: string,
  docName: string
): Promise<{ answer: string; sources: string[] }> {
  const systemPrompt = `你是一个基于文档知识的智能问答助手。
你的任务是：严格根据下方【参考文档】的内容回答用户问题。

规则：
1. 如果参考文档中有答案，请准确引用并总结
2. 如果参考文档中没有直接答案，请明确说"根据提供的文档，没有找到相关信息"
3. 回答要简洁、准确，不要编造文档中没有的内容
4. 如果适用，引用文档中的关键句子（用引号标注）`;

  const userMessage = `【参考文档】\n文档名称：${docName}\n\n${context}\n\n---\n用户提问：${question}\n\n请根据以上参考文档回答：`;

  const response = await openai.chat.completions.create({
    model: 'deepseek-chat',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage },
    ],
    temperature: 0.1,
    max_tokens: 1000,
  });

  const answer = response.choices[0].message.content || '无法生成回答。';

  // Extract relevant source snippets (first 120 chars of each context block, dedup'd)
  const sources = context
    .split(/\n---\n/)
    .filter(Boolean)
    .map((s) => s.trim().slice(0, 150) + (s.length > 150 ? '...' : ''));

  return { answer, sources };
}

// ============================================================
// 🌐 API Handler
// ============================================================
export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { action, docName } = body;

    // ─── 索引文档 ───
    if (action === 'index') {
      const { text, name } = body;
      if (!text || !name) {
        return NextResponse.json(
          { error: '缺少文本内容或文档名称' },
          { status: 400 }
        );
      }

      const chunks = chunkText(text, name);
      documentStore.set(name, chunks);

      return NextResponse.json({
        success: true,
        message: `文档 "${name}" 索引完成`,
        stats: {
          docName: name,
          chunkCount: chunks.length,
          totalChars: text.length,
          avgChunkSize: Math.round(
            chunks.reduce((sum, c) => sum + c.text.length, 0) / chunks.length
          ),
        },
      });
    }

    // ─── 查询文档 ───
    if (action === 'query') {
      const { question } = body;
      const targetDoc = docName || Array.from(documentStore.keys())[0];

      if (!question) {
        return NextResponse.json({ error: '缺少查询问题' }, { status: 400 });
      }

      if (!targetDoc || !documentStore.has(targetDoc)) {
        return NextResponse.json(
          { error: '没有已索引的文档。请先上传文档。' },
          { status: 400 }
        );
      }

      const chunks = documentStore.get(targetDoc)!;

      // Step 1: Retrieve relevant chunks
      const relevant = searchChunks(question, chunks, 5);

      if (relevant.length === 0) {
        return NextResponse.json({
          answer: '没有找到与问题相关的文档内容。请尝试换个问法。',
          sources: [],
          stats: { retrievedChunks: 0, totalChunks: chunks.length },
        });
      }

      // Step 2: Build context
      const context = relevant
        .map((c, i) => `[片段${i + 1}] ${c.text}`)
        .join('\n---\n');

      // Step 3: Generate answer with DeepSeek
      const { answer, sources } = await generateAnswer(
        question,
        context,
        targetDoc
      );

      return NextResponse.json({
        answer,
        sources,
        stats: {
          retrievedChunks: relevant.length,
          totalChunks: chunks.length,
          docName: targetDoc,
        },
      });
    }

    // ─── 清空文档库 ───
    if (action === 'clear') {
      const name = docName;
      if (name) {
        documentStore.delete(name);
      } else {
        documentStore.clear();
      }
      return NextResponse.json({
        success: true,
        message: name ? `文档 "${name}" 已清除` : '全部文档已清除',
      });
    }

    // ─── 查看状态 ───
    if (action === 'status') {
      const docs = Array.from(documentStore.entries()).map(([name, chunks]) => ({
        name,
        chunks: chunks.length,
        totalChars: chunks.reduce((sum, c) => sum + c.text.length, 0),
      }));

      return NextResponse.json({
        docCount: documentStore.size,
        documents: docs,
      });
    }

    return NextResponse.json(
      { error: `未知操作: ${action}。支持: index | query | clear | status` },
      { status: 400 }
    );
  } catch (error: any) {
    console.error('RAG 引擎异常:', error);
    return NextResponse.json(
      { error: `RAG 引擎异常: ${error.message}` },
      { status: 500 }
    );
  }
}
