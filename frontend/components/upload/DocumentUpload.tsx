'use client';

import { useState, useCallback } from 'react';
import { X, Upload, CheckCircle, AlertCircle, FileText, Loader2 } from 'lucide-react';

interface UploadResult {
    filename: string;
    status: 'success' | 'error';
    chunks?: number;
    time?: number;
    error?: string;
}

interface DocumentUploadProps {
    onClose: () => void;
    apiUrl: string;
}

export default function DocumentUpload({ onClose, apiUrl }: DocumentUploadProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [results, setResults] = useState<UploadResult[]>([]);
    const [tags, setTags] = useState('');

    const handleUpload = useCallback(async (files: FileList | File[]) => {
        setUploading(true);
        const fileArray = Array.from(files);

        for (const file of fileArray) {
            try {
                const formData = new FormData();
                formData.append('files', file);
                if (tags.trim()) formData.append('tags', tags.trim());

                const res = await fetch(`${apiUrl}/api/ingest`, {
                    method: 'POST',
                    body: formData,
                });

                const data = await res.json();

                if (res.ok && Array.isArray(data)) {
                    // API returns array of IngestionResult
                    const result = data[0];
                    if (result && result.status === 'completed') {
                        setResults((prev) => [...prev, {
                            filename: file.name,
                            status: 'success',
                            chunks: result.chunks_created,
                            time: result.processing_time_ms,
                        }]);
                    } else {
                        setResults((prev) => [...prev, {
                            filename: file.name,
                            status: 'error',
                            error: result?.error_message || 'Ingestion failed',
                        }]);
                    }
                } else {
                    // Non-200 or unexpected response
                    const errMsg = typeof data?.detail === 'string'
                        ? data.detail
                        : Array.isArray(data?.detail)
                            ? data.detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
                            : data?.error_message || 'Upload failed';
                    setResults((prev) => [...prev, {
                        filename: file.name,
                        status: 'error',
                        error: errMsg,
                    }]);
                }
            } catch (err: any) {
                setResults((prev) => [...prev, {
                    filename: file.name,
                    status: 'error',
                    error: err.message || 'Network error',
                }]);
            }
        }
        setUploading(false);
    }, [apiUrl, tags]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files.length > 0) handleUpload(e.dataTransfer.files);
    }, [handleUpload]);

    return (
        <div className="fixed inset-0 z-50 glass-modal flex items-center justify-center p-4"
            onClick={(e: React.MouseEvent) => e.target === e.currentTarget && onClose()}>
            <div className="bg-[#2f2f2f] rounded-2xl w-full max-w-lg border border-[#444] shadow-2xl animate-fade-in">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-[#444]">
                    <h2 className="text-base font-semibold text-white">Upload Documents</h2>
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded-lg hover:bg-[#444] text-[#8e8e8e] hover:text-white transition-colors"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6">
                    {/* Drop Zone */}
                    <div
                        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                        onDragLeave={() => setIsDragging(false)}
                        onDrop={handleDrop}
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${isDragging
                            ? 'border-[#10a37f] bg-[#10a37f]/5'
                            : 'border-[#555] hover:border-[#888]'
                            }`}
                        onClick={() => {
                            const input = document.createElement('input');
                            input.type = 'file';
                            input.multiple = true;
                            input.accept = '.txt,.md,.pdf,.csv,.json';
                            input.onchange = (e: any) => {
                                if (e.target.files) handleUpload(e.target.files);
                            };
                            input.click();
                        }}
                    >
                        <Upload className={`w-8 h-8 mx-auto mb-3 ${isDragging ? 'text-[#10a37f]' : 'text-[#666]'}`} />
                        <p className="text-sm text-[#ececec] mb-1">
                            Drop files here or <span className="text-[#10a37f]">browse</span>
                        </p>
                        <p className="text-xs text-[#666]">
                            TXT, MD, PDF, CSV, JSON — max 50MB
                        </p>
                    </div>

                    {/* Tags */}
                    <input
                        type="text"
                        value={tags}
                        onChange={(e) => setTags(e.target.value)}
                        placeholder="Tags (comma-separated, optional)"
                        className="mt-4 w-full bg-[#212121] border border-[#444] rounded-lg px-3 py-2.5 text-sm text-[#ececec] placeholder-[#666] focus:outline-none focus:border-[#10a37f] transition-colors"
                    />

                    {/* Upload Progress */}
                    {uploading && (
                        <div className="mt-4 flex items-center gap-2 text-sm text-[#b4b4b4]">
                            <Loader2 size={16} className="animate-spin text-[#10a37f]" />
                            Processing...
                        </div>
                    )}

                    {/* Results */}
                    {results.length > 0 && (
                        <div className="mt-4 space-y-2">
                            <div className="text-xs text-[#8e8e8e] font-medium mb-2">Results</div>
                            {results.map((r, i) => (
                                <div
                                    key={i}
                                    className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm ${r.status === 'success'
                                        ? 'bg-[#10a37f]/10 text-[#10a37f]'
                                        : 'bg-red-500/10 text-red-400'
                                        }`}
                                >
                                    {r.status === 'success' ? (
                                        <CheckCircle size={16} />
                                    ) : (
                                        <AlertCircle size={16} />
                                    )}
                                    <FileText size={14} className="opacity-60" />
                                    <span className="flex-1 truncate">{r.filename}</span>
                                    <span className="text-xs opacity-70">
                                        {r.status === 'success'
                                            ? `${r.chunks} chunks • ${r.time?.toFixed(0)}ms`
                                            : r.error}
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
