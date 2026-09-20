const STORY_CONFIRMATION_PATTERNS = [
    /想听什么(?:样|类型)?的?故事/,
    /什么样的故事/,
    /要不要听(?:一个)?故事/,
    /喜欢听什么(?:样|类型)?的?故事/,
    /给你讲(?:一个)?(?:有趣的)?故事(?:哦|吗|呢)?[?？]?$/,
];

export function taskContinuationForAssistantText(text) {
    const normalized = String(text || '').replace(/\s+/g, ' ').trim();
    if (!normalized.includes('故事')) return null;
    if (!STORY_CONFIRMATION_PATTERNS.some(pattern => pattern.test(normalized))) return null;
    return '现在直接开始讲故事，不要再确认或反问，用三至五句话完整讲完。';
}
