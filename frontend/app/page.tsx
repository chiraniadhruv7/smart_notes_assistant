'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import ChatInput from '@/components/chat/ChatInput';
import MessageBubble from '@/components/chat/MessageBubble';
import Sidebar from '@/components/sidebar/Sidebar';
import DocumentUpload from '@/components/upload/DocumentUpload';
import { Menu, Plus, Sparkles } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    citations?: any[];
    isStreaming?: boolean;
    timestamp: Date;
}

interface Conversation {
    id: string;
    title: string;
    messages: Message[];
    createdAt: Date;
}

export default function Home() {
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [showSidebar, setShowSidebar] = useState(true);
    const [showUpload, setShowUpload] = useState(false);
    const [apiStatus, setApiStatus] = useState<'connected' | 'disconnected'>('disconnected');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const activeConversation = conversations.find((c: Conversation) => c.id === activeConversationId);
    const messages = activeConversation?.messages || [];

    useEffect(() => {
        const check = async () => {
            try {
                const res = await fetch(`${API_URL}/health`);
                setApiStatus(res.ok ? 'connected' : 'disconnected');
            } catch (_e) {
                setApiStatus('disconnected');
            }
        };
        check();
        const interval = setInterval(check, 15000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const createNewConversation = useCallback(() => {
        const newConvo: Conversation = {
            id: uuidv4(),
            title: 'New chat',
            messages: [],
            createdAt: new Date(),
        };
        setConversations((prev: Conversation[]) => [newConvo, ...prev]);
        setActiveConversationId(newConvo.id);
        return newConvo.id;
    }, []);

    const handleSend = useCallback(async (query: string) => {
        if (!query.trim() || isLoading) return;

        let convoId = activeConversationId;
        if (!convoId) {
            convoId = createNewConversation();
        }

        const userMsg: Message = {
            id: uuidv4(),
            role: 'user',
            content: query,
            timestamp: new Date(),
        };

        const assistantMsgId = uuidv4();

        setConversations((prev: Conversation[]) =>
            prev.map((c: Conversation) => {
                if (c.id === convoId) {
                    const isFirst = c.messages.length === 0;
                    return {
                        ...c,
                        title: isFirst ? query.slice(0, 40) + (query.length > 40 ? '...' : '') : c.title,
                        messages: [
                            ...c.messages,
                            userMsg,
                            { id: assistantMsgId, role: 'assistant' as const, content: '', isStreaming: true, timestamp: new Date() },
                        ],
                    };
                }
                return c;
            })
        );

        setIsLoading(true);

        try {
            const res = await fetch(`${API_URL}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, conversation_id: convoId, stream: true }),
            });

            if (!res.ok) {
                throw new Error(`API returned ${res.status}`);
            }

            const reader = res.body?.getReader();
            const decoder = new TextDecoder();
            if (!reader) throw new Error('No stream');

            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (let idx = 0; idx < lines.length; idx++) {
                    const line = lines[idx];
                    if (!line.startsWith('data: ')) continue;

                    try {
                        const data = JSON.parse(line.slice(6));
                        const prevLine = idx > 0 ? lines[idx - 1] : '';
                        const eventType = prevLine.startsWith('event: ') ? prevLine.slice(7).trim() : 'token';

                        if (eventType === 'token' && typeof data?.token === 'string') {
                            const token = data.token;
                            setConversations((prev: Conversation[]) =>
                                prev.map((c: Conversation) =>
                                    c.id === convoId
                                        ? { ...c, messages: c.messages.map((m: Message) => m.id === assistantMsgId ? { ...m, content: m.content + token } : m) }
                                        : c
                                )
                            );
                        } else if (eventType === 'citations' && Array.isArray(data)) {
                            const safeCits = data.map((ci: any) => ({
                                document_name: String(ci.document_name || ''),
                                chunk_id: String(ci.chunk_id || ''),
                                content: String(ci.content || ''),
                                relevance_score: Number(ci.relevance_score || 0),
                            }));
                            setConversations((prev: Conversation[]) =>
                                prev.map((c: Conversation) =>
                                    c.id === convoId
                                        ? { ...c, messages: c.messages.map((m: Message) => m.id === assistantMsgId ? { ...m, citations: safeCits } : m) }
                                        : c
                                )
                            );
                        } else if (eventType === 'done') {
                            setConversations((prev: Conversation[]) =>
                                prev.map((c: Conversation) =>
                                    c.id === convoId
                                        ? { ...c, messages: c.messages.map((m: Message) => m.id === assistantMsgId ? { ...m, isStreaming: false } : m) }
                                        : c
                                )
                            );
                        }
                    } catch (_parseErr) {
                        // Skip unparseable SSE lines
                    }
                }
            }

            // Mark streaming done after loop
            setConversations((prev: Conversation[]) =>
                prev.map((c: Conversation) =>
                    c.id === convoId
                        ? { ...c, messages: c.messages.map((m: Message) => m.id === assistantMsgId ? { ...m, isStreaming: false } : m) }
                        : c
                )
            );
        } catch (_err) {
            setConversations((prev: Conversation[]) =>
                prev.map((c: Conversation) =>
                    c.id === convoId
                        ? { ...c, messages: c.messages.map((m: Message) => m.id === assistantMsgId ? { ...m, content: 'Sorry, something went wrong. Please try again.', isStreaming: false } : m) }
                        : c
                )
            );
        } finally {
            setIsLoading(false);
        }
    }, [activeConversationId, isLoading, createNewConversation]);

    const deleteConversation = useCallback((id: string) => {
        setConversations((prev: Conversation[]) => prev.filter((c: Conversation) => c.id !== id));
        if (activeConversationId === id) setActiveConversationId(null);
    }, [activeConversationId]);

    return (
        <div className="flex h-screen overflow-hidden">
            <div className={`${showSidebar ? 'w-[260px]' : 'w-0'} transition-all duration-300 flex-shrink-0 overflow-hidden`}>
                <Sidebar
                    conversations={conversations}
                    activeConversationId={activeConversationId}
                    onSelectConversation={setActiveConversationId}
                    onNewConversation={createNewConversation}
                    onDeleteConversation={deleteConversation}
                    onOpenUpload={() => setShowUpload(true)}
                />
            </div>

            <div className="flex-1 flex flex-col min-w-0 bg-[#212121]">
                <header className="h-12 flex items-center px-3 flex-shrink-0">
                    <button onClick={() => setShowSidebar(!showSidebar)} className="p-2 rounded-lg hover:bg-[#2f2f2f] text-[#b4b4b4] hover:text-white transition-colors">
                        <Menu size={18} />
                    </button>
                    <button onClick={createNewConversation} className="p-2 rounded-lg hover:bg-[#2f2f2f] text-[#b4b4b4] hover:text-white transition-colors ml-1">
                        <Plus size={18} />
                    </button>
                    <div className="flex-1 flex justify-center">
                        <span className="text-sm font-medium text-[#b4b4b4]">Smart Notes Assistant</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${apiStatus === 'connected' ? 'bg-[#10a37f]' : 'bg-red-500'}`} />
                    </div>
                </header>

                <div className="flex-1 overflow-y-auto">
                    {messages.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center px-4 animate-fade-in">
                            <div className="mb-8">
                                <div className="w-16 h-16 rounded-2xl bg-[#10a37f] flex items-center justify-center shadow-lg shadow-[#10a37f]/20">
                                    <Sparkles className="w-8 h-8 text-white" />
                                </div>
                            </div>
                            <h1 className="text-2xl font-semibold text-white mb-2">Smart Notes Assistant</h1>
                            <p className="text-[#8e8e8e] text-sm mb-10 text-center max-w-md">
                                Upload documents and ask questions. I&apos;ll find answers using intelligent retrieval and AI.
                            </p>
                            <div className="grid grid-cols-2 gap-3 max-w-xl w-full">
                                <button onClick={() => handleSend('Summarize a document')} className="text-left p-4 rounded-xl border border-[#444] hover:bg-[#2f2f2f] transition-all duration-200">
                                    <span className="text-lg mb-2 block">📄</span>
                                    <div className="text-sm text-white font-medium">Summarize a document</div>
                                    <div className="text-xs text-[#8e8e8e] mt-1">Upload and get key insights</div>
                                </button>
                                <button onClick={() => handleSend('Search knowledge base')} className="text-left p-4 rounded-xl border border-[#444] hover:bg-[#2f2f2f] transition-all duration-200">
                                    <span className="text-lg mb-2 block">🔍</span>
                                    <div className="text-sm text-white font-medium">Search knowledge base</div>
                                    <div className="text-xs text-[#8e8e8e] mt-1">Find answers from your docs</div>
                                </button>
                                <button onClick={() => handleSend('Generate questions')} className="text-left p-4 rounded-xl border border-[#444] hover:bg-[#2f2f2f] transition-all duration-200">
                                    <span className="text-lg mb-2 block">💡</span>
                                    <div className="text-sm text-white font-medium">Generate questions</div>
                                    <div className="text-xs text-[#8e8e8e] mt-1">Create study material from PDFs</div>
                                </button>
                                <button onClick={() => handleSend('Analyze content')} className="text-left p-4 rounded-xl border border-[#444] hover:bg-[#2f2f2f] transition-all duration-200">
                                    <span className="text-lg mb-2 block">📊</span>
                                    <div className="text-sm text-white font-medium">Analyze content</div>
                                    <div className="text-xs text-[#8e8e8e] mt-1">Deep dive into uploaded files</div>
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="max-w-3xl mx-auto w-full px-4 py-4">
                            {messages.map((msg: Message) => (
                                <MessageBubble key={msg.id} message={msg} />
                            ))}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                <div className="flex-shrink-0 pb-4 px-4">
                    <div className="max-w-3xl mx-auto">
                        <ChatInput onSend={handleSend} isLoading={isLoading} onUploadClick={() => setShowUpload(true)} />
                        <p className="text-[10px] text-[#8e8e8e] text-center mt-2">
                            Smart Notes Assistant retrieves answers from your uploaded documents using AI.
                        </p>
                    </div>
                </div>
            </div>

            {showUpload && <DocumentUpload onClose={() => setShowUpload(false)} apiUrl={API_URL} />}
        </div>
    );
}
