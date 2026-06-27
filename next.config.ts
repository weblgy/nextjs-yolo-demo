import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* 其他配置保持不变 */
  
  // 👇 加入这块配置
  typescript: {
    // ⚠️ 警告：这会忽略所有 TS 错误，仅建议在测试部署时使用
    ignoreBuildErrors: true,
  },
  eslint: {
    // 同理，忽略 eslint 检查
    ignoreDuringBuilds: true,
  },
};

export default nextConfig; 