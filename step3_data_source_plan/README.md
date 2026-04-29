# Step 3 Data Source Plan

## Source

The text source is the Management's Discussion and Analysis section of public company 10-K annual reports from SEC EDGAR. MD&A is usually Item 7 in a 10-K and contains management-written discussion of operations, strategy, investments, market conditions, risks, and performance.

## Fit With Business Purpose

The project construct is strategic orientation in firm language: exploration, exploitation, ambidextrous, or neither. MD&A fits because firms use it to explain how they are growing, improving existing operations, entering markets, managing uncertainty, and allocating resources. Those statements are directly relevant to detecting exploratory and exploitative behavior from text.

## Regular Firm Use

A firm, investor relations team, strategy group, or competitive intelligence team could monitor peer 10-K MD&A language annually after filings are released, with quarterly refreshes using 10-Q text if the system is extended. Results could support peer benchmarking, board reporting, strategy dashboards, and trend monitoring by industry or competitor.

## Access Plan

The collection process uses SEC EDGAR public endpoints:

- Ticker to CIK lookup from SEC's company ticker mapping.
- 10-K filing metadata from `https://data.sec.gov/submissions/CIK##########.json`.
- Filing HTML from the official SEC Archives document URL.

The script identifies itself with a `User-Agent`, spaces requests with a delay, and downloads only public filings. Do not bypass SEC rate limits or site rules.

## Sampling Approach

The sampling frame starts with public companies across eight broad industries. For each firm, the collection script downloads recent 10-K filings, extracts MD&A text, splits it into sentence-level examples, and caps per-company volume to avoid domination by a few firms. Step 5 locks a 1,000-sentence holdout and creates a 15,000-sentence train/test pool while spreading samples across industry, company, and filing year.
