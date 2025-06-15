-- Supabase Database Schema for Cohere Hybrid RAG Platform
-- Run these SQL commands in your Supabase SQL Editor

-- Table for storing chat history
CREATE TABLE IF NOT EXISTS chat_history (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    dataset_id TEXT,
    session_id UUID,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for storing user dataset metadata
CREATE TABLE IF NOT EXISTS user_datasets (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    dataset_id TEXT NOT NULL,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL, -- 'cohere', 'local_upload'
    file_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, dataset_id)
);

-- Table for storing user preferences and settings
CREATE TABLE IF NOT EXISTS user_preferences (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL UNIQUE,
    cohere_api_key_encrypted TEXT,
    preferred_model TEXT DEFAULT 'command-r-plus',
    default_temperature DECIMAL(3,2) DEFAULT 0.3,
    max_tokens INTEGER DEFAULT 800,
    preferred_mode TEXT DEFAULT 'hybrid', -- 'hybrid', 'cohere_only', 'local_only'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for tracking embedding jobs
CREATE TABLE IF NOT EXISTS embed_jobs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    job_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    job_name TEXT NOT NULL,
    status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp ON chat_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_datasets_user_id ON user_datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_embed_jobs_user_id ON embed_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_embed_jobs_status ON embed_jobs(status);

-- Row Level Security (RLS) policies
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE embed_jobs ENABLE ROW LEVEL SECURITY;

-- Policies for chat_history
CREATE POLICY "Users can view their own chat history" ON chat_history
    FOR SELECT USING (user_id = auth.uid()::uuid);

CREATE POLICY "Users can insert their own chat history" ON chat_history
    FOR INSERT WITH CHECK (user_id = auth.uid()::uuid);

-- Policies for user_datasets
CREATE POLICY "Users can view their own datasets" ON user_datasets
    FOR SELECT USING (user_id = auth.uid()::uuid);

CREATE POLICY "Users can insert their own datasets" ON user_datasets
    FOR INSERT WITH CHECK (user_id = auth.uid()::uuid);

-- Policies for user_preferences
CREATE POLICY "Users can view their own preferences" ON user_preferences
    FOR SELECT USING (user_id = auth.uid()::uuid);

CREATE POLICY "Users can update their own preferences" ON user_preferences
    FOR ALL USING (user_id = auth.uid()::uuid);

-- Policies for embed_jobs
CREATE POLICY "Users can view their own embed jobs" ON embed_jobs
    FOR SELECT USING (user_id = auth.uid()::uuid);

CREATE POLICY "Users can insert their own embed jobs" ON embed_jobs
    FOR INSERT WITH CHECK (user_id = auth.uid()::uuid);