import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Github, Camera } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
      <div className="container flex h-16 items-center justify-between px-4 md:px-8">
        <Link href="/" className="flex items-center gap-2 font-bold text-xl">
          <Camera className="h-6 w-6" />
          <span>AI Lab</span>
        </Link>
        
        <div className="flex gap-6 text-sm font-medium items-center">
          <Link href="/" className="hover:text-primary transition-colors">Home</Link>
          <Link href="/demo" className="hover:text-primary transition-colors">YOLO</Link>
          <Link href="/blog" className="hover:text-primary transition-colors">Blog</Link>
        </div>

        <div className="flex items-center gap-2">
          <Link href="https://github.com/weblgy" target="_blank">
            <Button variant="outline" size="icon">
              <Github className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>
    </nav>
  );
}