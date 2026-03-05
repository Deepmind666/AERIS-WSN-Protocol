---
name: citation-audit
description: Audit BibTeX references for accuracy. Use when verifying DOIs, checking for hallucinated references, finding missing citations, or validating reference metadata against CrossRef/arXiv.
metadata:
  category: academic-writing
  tags: [bibtex, references, doi, crossref, arxiv, citation, verification]
argument-hint: "[references.bib] [--mode full|quick|doi-only]"
allowed-tools: Read, Glob, Grep, Bash(python *), Bash(curl *), WebFetch, WebSearch
---

# Citation Audit Skill

Verify BibTeX references for accuracy, completeness, and consistency.

## When to Use

- Before journal submission: full audit of all references
- After adding new citations: quick check of new entries
- Reviewer asks about reference accuracy: targeted verification
- Detect hallucinated/fabricated references

## Audit Workflow

### Phase 1: Parse and Inventory
1. Read the `.bib` file specified in `$ARGUMENTS` (default: find `*.bib` in project)
2. Extract all entries: count by type (@article, @inproceedings, @book, @misc)
3. List entries missing DOI field
4. List entries with arXiv IDs (check if formally published)

### Phase 2: CrossRef Verification (for entries with DOI)
For each entry with a DOI:
1. Query CrossRef API: `curl -s "https://api.crossref.org/works/{DOI}"`
2. Compare returned metadata against BibTeX fields:
   - Title match (fuzzy, ignore case)
   - Author last names match
   - Year match
   - Journal/booktitle match
3. Flag mismatches as **P0** (title/author wrong = possible hallucination) or **P1** (year/journal minor discrepancy)

### Phase 3: Completeness Check
For each entry, verify required fields by type:
- `@article`: author, title, journal, year, volume (pages recommended)
- `@inproceedings`: author, title, booktitle, year
- `@book`: author/editor, title, publisher, year
- `@misc`: author, title, year, note or howpublished

### Phase 4: arXiv Update Check
For entries with `archiveprefix = {arXiv}` or URL containing `arxiv.org`:
1. Search CrossRef/Google Scholar for the title
2. If a formal publication exists, flag as **P1**: "arXiv preprint has been formally published, update entry"

### Phase 5: Duplicate Detection
- Check for duplicate DOIs
- Check for entries with same title (fuzzy match)
- Check for entries with same first author + year + similar title

## Output Format

Generate `ref_audit_report.csv` with columns:
```
citation_key, severity, category, issue, evidence
```

Severity levels:
- **P0**: Possible hallucination (title/author mismatch with DOI)
- **P1**: Missing/incorrect metadata (no DOI, wrong year, arXiv→published)
- **P2**: Style issue (inconsistent journal abbreviation, missing pages)

## Constraints
- Do NOT modify the .bib file without explicit user permission
- Report findings only; let user decide what to fix
- If CrossRef API is rate-limited, use 1-second delays between requests
- For entries without DOI, use title-based web search as fallback
