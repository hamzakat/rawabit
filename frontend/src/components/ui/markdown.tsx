import { Fragment, memo, useMemo, type ReactNode } from "react";

import { cn } from "@/lib/utils";

type MarkdownBlock =
  | { type: "heading"; level: 1 | 2 | 3 | 4 | 5 | 6; text: string }
  | { type: "paragraph"; text: string }
  | { type: "unordered-list"; items: string[] }
  | { type: "ordered-list"; items: string[] }
  | { type: "blockquote"; text: string }
  | { type: "code"; language?: string; code: string };

const INLINE_PATTERN =
  /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*/g;
const MAX_INLINE_SOURCE_LENGTH = 4000;
const MAX_INLINE_TOKENS = 400;

function isBlank(value: string) {
  return value.trim().length === 0;
}

function isOrderedListLine(value: string) {
  return /^\d+\.\s+/.test(value);
}

function isUnorderedListLine(value: string) {
  return /^[-*+]\s+/.test(value);
}

function consumeListBlock(
  lines: string[],
  startIndex: number,
  matcher: (value: string) => boolean,
  stripPrefix: (value: string) => string
) {
  const items: string[] = [];
  let index = startIndex;

  while (index < lines.length) {
    const current = lines[index];
    if (isBlank(current)) {
      let lookahead = index + 1;
      while (lookahead < lines.length && isBlank(lines[lookahead])) {
        lookahead += 1;
      }
      if (lookahead >= lines.length || !matcher(lines[lookahead])) {
        break;
      }
      index = lookahead;
      continue;
    }

    if (!matcher(current)) {
      break;
    }

    items.push(stripPrefix(current).trim());
    index += 1;
  }

  return { items, nextIndex: index };
}

function parseInline(text: string): ReactNode[] {
  if (text.length === 0) {
    return [];
  }
  if (text.length > MAX_INLINE_SOURCE_LENGTH) {
    return [text];
  }

  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  INLINE_PATTERN.lastIndex = 0;
  match = INLINE_PATTERN.exec(text);
  while (match && key < MAX_INLINE_TOKENS) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }

    if (match[1] && match[2]) {
      nodes.push(
        <a
          key={`link-${key}`}
          href={match[2]}
          target="_blank"
          rel="noreferrer"
          className="font-medium text-primary underline underline-offset-4"
        >
          {match[1]}
        </a>
      );
    } else if (match[3]) {
      nodes.push(
        <code
          key={`code-${key}`}
          className="rounded bg-muted px-1 py-0.5 font-mono text-[0.92em] text-foreground"
        >
          {match[3]}
        </code>
      );
    } else if (match[4]) {
      nodes.push(
        <strong key={`strong-${key}`} className="font-semibold text-foreground">
          {match[4]}
        </strong>
      );
    } else if (match[5]) {
      nodes.push(
        <em key={`em-${key}`} className="italic">
          {match[5]}
        </em>
      );
    }

    lastIndex = INLINE_PATTERN.lastIndex;
    key += 1;
    match = INLINE_PATTERN.exec(text);
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}

function parseBlocks(content: string): MarkdownBlock[] {
  const normalized = content.replace(/\r\n?/g, "\n");
  const lines = normalized.split("\n");
  const blocks: MarkdownBlock[] = [];

  let index = 0;
  while (index < lines.length) {
    const line = lines[index];

    if (isBlank(line)) {
      index += 1;
      continue;
    }

    if (line.startsWith("```")) {
      const language = line.slice(3).trim() || undefined;
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length && lines[index].startsWith("```")) {
        index += 1;
      }
      blocks.push({ type: "code", language, code: codeLines.join("\n") });
      continue;
    }

    const headingMatch = /^(#{1,6})\s+(.*)$/.exec(line);
    if (headingMatch) {
      blocks.push({
        type: "heading",
        level: headingMatch[1].length as 1 | 2 | 3 | 4 | 5 | 6,
        text: headingMatch[2].trim()
      });
      index += 1;
      continue;
    }

    if (isUnorderedListLine(line)) {
      const { items, nextIndex } = consumeListBlock(
        lines,
        index,
        isUnorderedListLine,
        (value) => value.replace(/^[-*+]\s+/, "")
      );
      index = nextIndex;
      blocks.push({ type: "unordered-list", items });
      continue;
    }

    if (isOrderedListLine(line)) {
      const { items, nextIndex } = consumeListBlock(
        lines,
        index,
        isOrderedListLine,
        (value) => value.replace(/^\d+\.\s+/, "")
      );
      index = nextIndex;
      blocks.push({ type: "ordered-list", items });
      continue;
    }

    if (line.startsWith(">")) {
      const quoteLines: string[] = [];
      while (index < lines.length && lines[index].startsWith(">")) {
        quoteLines.push(lines[index].replace(/^>\s?/, "").trim());
        index += 1;
      }
      blocks.push({ type: "blockquote", text: quoteLines.join(" ") });
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length) {
      const candidate = lines[index];
      if (
        isBlank(candidate) ||
        candidate.startsWith("```") ||
        /^(#{1,6})\s+/.test(candidate) ||
        isUnorderedListLine(candidate) ||
        isOrderedListLine(candidate) ||
        candidate.startsWith(">")
      ) {
        break;
      }
      paragraphLines.push(candidate.trim());
      index += 1;
    }
    blocks.push({
      type: "paragraph",
      text: paragraphLines.join(" ")
    });
  }

  return blocks;
}

function headingClassName(level: 1 | 2 | 3 | 4 | 5 | 6) {
  if (level === 1) {
    return "text-base font-semibold text-foreground";
  }
  if (level === 2) {
    return "text-sm font-semibold text-foreground";
  }
  return "text-sm font-medium text-foreground";
}

export const MarkdownText = memo(function MarkdownText({
  content,
  className
}: {
  content: string;
  className?: string;
}) {
  const blocks = useMemo(() => parseBlocks(content), [content]);
  const renderedBlocks = useMemo(
    () =>
      blocks.map((block, index) => {
        if (block.type === "heading") {
          return (
            <p key={`heading-${index}`} className={headingClassName(block.level)}>
              {parseInline(block.text)}
            </p>
          );
        }
        if (block.type === "unordered-list") {
          return (
            <ul key={`ul-${index}`} className="ml-5 list-disc space-y-1">
              {block.items.map((item, itemIndex) => (
                <li key={`ul-item-${index}-${itemIndex}`}>{parseInline(item)}</li>
              ))}
            </ul>
          );
        }
        if (block.type === "ordered-list") {
          return (
            <ol key={`ol-${index}`} className="ml-5 list-decimal space-y-1">
              {block.items.map((item, itemIndex) => (
                <li key={`ol-item-${index}-${itemIndex}`}>{parseInline(item)}</li>
              ))}
            </ol>
          );
        }
        if (block.type === "blockquote") {
          return (
            <blockquote
              key={`quote-${index}`}
              className="border-l-2 border-border/70 pl-3 italic text-muted-foreground"
            >
              {parseInline(block.text)}
            </blockquote>
          );
        }
        if (block.type === "code") {
          return (
            <div key={`code-${index}`} className="rounded-lg border bg-background/80">
              {block.language ? (
                <div className="border-b px-3 py-1.5 text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
                  {block.language}
                </div>
              ) : null}
              <pre className="overflow-x-auto px-3 py-2 text-xs text-foreground">
                <code>{block.code}</code>
              </pre>
            </div>
          );
        }
        return (
          <p key={`paragraph-${index}`} className="break-words text-sm text-foreground">
            {parseInline(block.text).map((node, nodeIndex) => (
              <Fragment key={`paragraph-node-${index}-${nodeIndex}`}>{node}</Fragment>
            ))}
          </p>
        );
      }),
    [blocks]
  );

  return (
    <div className={cn("space-y-3 text-sm leading-relaxed text-foreground", className)}>
      {renderedBlocks}
    </div>
  );
});
