'use client';

import { User, Sparkles, ChevronDown, FileText } from 'lucide-react';
import { useState } from 'react';

interface Citation {
    document_name: string;
    chunk_id: string;
    content: string;
    relevance_score: number;
    metadata?: Record<string, string>;
}

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    citations?: Citation[];
    isStreaming?: boolean;
    timestamp: Date;
}

function MarkdownContent({ content }: { content: string }) {
    // Simple markdown-like renderer without react-markdown dependency
    const lines = content.split('\n');
    const elements: React.ReactNode[] = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        if (line.startsWith('### ')) {
            elements.push(<h3 key={i} className="font-semibold text-white mt-4 mb-2 text-base">{line.slice(4)}</h3>);
        } else if (line.startsWith('## ')) {
            elements.push(<h2 key={i} className="font-semibold text-white mt-5 mb-2 text-lg">{line.slice(3)}</h2>);
        } else if (line.startsWith('# ')) {
            elements.push(<h1 key={i} className="font-semibold text-white mt-6 mb-3 text-xl">{line.slice(2)}</h1>);
        } else if (line.startsWith('- ') || line.startsWith('* ')) {
            elements.push(
                <div key={i} className="flex gap-2 ml-4 mb-1">
                    <span className="text-[#8e8e8e] mt-1">•</span>
                    <span>{formatInline(line.slice(2))}</span>
                </div>
            );
        } else if (line.startsWith('```')) {
            // Code block: collect lines until closing ```
            const codeLines: string[] = [];
            i++;
            while (i < lines.length && !lines[i].startsWith('```')) {
                codeLines.push(lines[i]);
                i++;
            }
            elements.push(
                <pre key={`code-${i}`} className="bg-[#1a1a1a] rounded-xl p-4 my-3 overflow-x-auto border border-[#444] text-[13px] font-mono text-[#ececec]">
                    <code>{codeLines.join('\n')}</code>
                </pre>
            );
        } else if (line.trim() === '') {
            elements.push(<div key={i} className="h-3" />);
        } else {
            elements.push(<p key={i} className="mb-2 leading-7">{formatInline(line)}</p>);
        }
    }

    return <>{elements}</>;
}

function formatInline(text: string): React.ReactNode {
    // Handle **bold** and `code` inline formatting
    const parts: React.ReactNode[] = [];
    const regex = /(\*\*(.+?)\*\*|`(.+?)`)/g;
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
        if (match.index > lastIndex) {
            parts.push(text.slice(lastIndex, match.index));
        }
        if (match[2]) {
            parts.push(<strong key={match.index} className="text-white font-semibold">{match[2]}</strong>);
        } else if (match[3]) {
            parts.push(<code key={match.index} className="bg-[#1a1a1a] text-[#e06c75] px-1.5 py-0.5 rounded text-[13px] font-mono">{match[3]}</code>);
        }
        lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex));
    }

    return parts.length > 0 ? parts : text;
}

export default function MessageBubble({ message }: { message: Message }) {
    const [showCitations, setShowCitations] = useState(false);
    const isUser = message.role === 'user';

    return (
        <div className="py-5 animate-fade-in">
            <div className="flex gap-4">
                {/* Avatar */}
                <div className="flex-shrink-0 mt-0.5">
                    {isUser ? (
                        <div className="w-7 h-7 rounded-full bg-[#5436DA] flex items-center justify-center">
                            <User size={14} className="text-white" />
                        </div>
                    ) : (
                        <div className="w-7 h-7 rounded-full bg-[#10a37f] flex items-center justify-center">
                            <Sparkles size={14} className="text-white" />
                        </div>
                    )}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-white mb-1.5">
                        {isUser ? 'You' : 'Assistant'}
                    </div>

                    {isUser ? (
                        <div className="text-[15px] text-[#ececec] leading-7 whitespace-pre-wrap">
                            {message.content}
                        </div>
                    ) : (
                        <div className="text-[15px] text-[#ececec] leading-7">
                            {message.content ? (
                                <MarkdownContent content={message.content} />
                            ) : message.isStreaming ? (
                                <div className="typing-indicator flex items-center gap-1 py-2">
                                    <span /><span /><span />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* Citations */}
                    {message.citations && message.citations.length > 0 && !message.isStreaming && (
                        <div className="mt-4">
                            <button
                                onClick={() => setShowCitations(!showCitations)}
                                className="flex items-center gap-2 text-xs text-[#8e8e8e] hover:text-[#b4b4b4] transition-colors"
                            >
                                <FileText size={13} />
                                <span>{message.citations.length} sources</span>
                                <ChevronDown
                                    size={13}
                                    className={`transition-transform duration-200 ${showCitations ? 'rotate-180' : ''}`}
                                />
                            </button>

                            {showCitations && (
                                <div className="mt-2 space-y-2 animate-fade-in">
                                    {message.citations.map((cit: Citation, i: number) => (
                                        <div
                                            key={i}
                                            className="border border-[#444] rounded-lg p-3 text-xs"
                                        >
                                            <div className="flex items-center justify-between mb-1.5">
                                                <span className="font-medium text-white flex items-center gap-1.5">
                                                    <span className="bg-[#10a37f] text-white text-[10px] w-5 h-5 rounded flex items-center justify-center font-bold">
                                                        {i + 1}
                                                    </span>
                                                    {cit.document_name}
                                                </span>
                                                <span className="text-[#10a37f]">
                                                    {(cit.relevance_score * 100).toFixed(0)}% match
                                                </span>
                                            </div>
                                            <p className="text-[#8e8e8e] line-clamp-2 leading-relaxed">
                                                {cit.content}
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
