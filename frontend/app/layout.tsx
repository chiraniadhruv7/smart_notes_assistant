import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
    title: 'Smart Notes Assistant',
    description: 'AI-powered knowledge assistant with intelligent document retrieval',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body className="bg-[#212121] text-[#ececec] font-sans antialiased min-h-screen"
                style={{ fontFamily: "'Inter', system-ui, -apple-system, sans-serif" }}>
                {children}
            </body>
        </html>
    );
}
