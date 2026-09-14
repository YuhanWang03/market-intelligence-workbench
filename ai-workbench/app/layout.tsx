import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AI Hedge Fund Workbench',
  description: 'AI-powered market monitoring, investment research, and strategy lab.',
  openGraph: {
    title: 'AI Hedge Fund Workbench',
    description: '盯盘 · 研究 · 实验室',
    images: [{ url: '/og.png', width: 1732, height: 910 }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'AI Hedge Fund Workbench',
    description: '盯盘 · 研究 · 实验室',
    images: ['/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body></html>;
}
