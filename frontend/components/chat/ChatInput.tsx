'use client';

import { useState, useRef, useEffect } from 'react';
import { ArrowUp, Paperclip, Square } from 'lucide-react';

interface ChatInputProps {
    onSend: (message: string) => void;
    isLoading: boolean;
    onUploadClick: () => void;
}

export default function ChatInput({ onSend, isLoading, onUploadClick }: ChatInputProps) {
    const [input, setInput] = useState('');
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Auto-resize textarea
    useEffect(() => {
        const ta = textareaRef.current;
        if (ta) {
            ta.style.height = 'auto';
            ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
        }
    }, [input]);

    const handleSubmit = () => {
        if (!input.trim() || isLoading) return;
        onSend(input.trim());
        setInput('');
        if (textareaRef.current) textareaRef.current.style.height = 'auto';
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="relative bg-[#2f2f2f] rounded-2xl border border-[#444] focus-within:border-[#666] transition-colors shadow-lg">
            <div className="flex items-end gap-2 p-3">
                {/* Upload button */}
                <button
                    onClick={onUploadClick}
                    className="p-1.5 rounded-lg hover:bg-[#444] text-[#8e8e8e] hover:text-white transition-colors flex-shrink-0 mb-0.5"
                    title="Upload document"
                >
                    <Paperclip size={18} />
                </button>

                {/* Textarea */}
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Message Smart Notes Assistant..."
                    rows={1}
                    className="flex-1 bg-transparent text-[#ececec] placeholder-[#8e8e8e] text-[15px] resize-none outline-none leading-6 max-h-[200px]"
                    style={{ fontFamily: "'Inter', system-ui, sans-serif" }}
                />

                {/* Send / Stop button */}
                <button
                    onClick={handleSubmit}
                    disabled={!input.trim() && !isLoading}
                    className={`p-1.5 rounded-lg flex-shrink-0 mb-0.5 transition-all duration-200 ${input.trim() || isLoading
                            ? 'bg-white text-[#212121] hover:bg-[#d9d9d9]'
                            : 'bg-[#444] text-[#8e8e8e] cursor-not-allowed'
                        }`}
                >
                    {isLoading ? <Square size={16} /> : <ArrowUp size={16} />}
                </button>
            </div>
        </div>
    );
}
