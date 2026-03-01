'use client';

import { useState } from 'react';
import { Plus, MessageSquare, Trash2, Upload } from 'lucide-react';

interface Conversation {
    id: string;
    title: string;
    messages: any[];
    createdAt: Date;
}

interface SidebarProps {
    conversations: Conversation[];
    activeConversationId: string | null;
    onSelectConversation: (id: string) => void;
    onNewConversation: () => void;
    onDeleteConversation: (id: string) => void;
    onOpenUpload: () => void;
}

export default function Sidebar({
    conversations,
    activeConversationId,
    onSelectConversation,
    onNewConversation,
    onDeleteConversation,
    onOpenUpload,
}: SidebarProps) {
    return (
        <div className="h-full flex flex-col bg-[#171717] w-[260px]">
            {/* Header */}
            <div className="p-3 flex flex-col gap-2">
                <button
                    onClick={onNewConversation}
                    className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-[#212121] text-[#ececec] text-sm transition-colors"
                >
                    <Plus size={16} />
                    <span>New chat</span>
                </button>
                <button
                    onClick={onOpenUpload}
                    className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-[#212121] text-[#b4b4b4] text-sm transition-colors"
                >
                    <Upload size={16} />
                    <span>Upload documents</span>
                </button>
            </div>

            {/* Conversations */}
            <div className="flex-1 overflow-y-auto px-2 pb-2">
                {conversations.length > 0 && (
                    <div className="px-2 py-2 text-xs text-[#8e8e8e] font-medium">Recent</div>
                )}
                {conversations.map((convo: Conversation) => (
                    <div
                        key={convo.id}
                        className={`group flex items-center rounded-lg px-3 py-2.5 mb-0.5 cursor-pointer transition-colors ${activeConversationId === convo.id
                                ? 'bg-[#212121] text-white'
                                : 'text-[#b4b4b4] hover:bg-[#212121] hover:text-[#ececec]'
                            }`}
                        onClick={() => onSelectConversation(convo.id)}
                    >
                        <MessageSquare size={14} className="flex-shrink-0 mr-3 opacity-60" />
                        <span className="text-sm truncate flex-1">{convo.title}</span>
                        <button
                            onClick={(e: React.MouseEvent) => {
                                e.stopPropagation();
                                onDeleteConversation(convo.id);
                            }}
                            className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-all"
                        >
                            <Trash2 size={13} />
                        </button>
                    </div>
                ))}
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-[#2f2f2f]">
                <div className="text-[11px] text-[#666] text-center">
                    Smart Notes Assistant v1.0
                </div>
            </div>
        </div>
    );
}
