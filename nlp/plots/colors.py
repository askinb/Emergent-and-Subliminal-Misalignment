"""Shared color palette for paper plots. Use these everywhere for consistency."""

# 4 unique colors for tasks (alphabetical order matches cfg.TASKS sorted).
TASK_COLORS = {
    "advice":         "#d62728",  # red
    "critique":       "#1f77b4",  # blue
    "summarization":  "#2ca02c",  # green
    "tutor":          "#9467bd",  # purple
}

# 3 unique colors for domains.
DOMAIN_COLORS = {
    "medical":  "#17becf",  # cyan/teal
    "sports":   "#ff7f0e",  # orange
    "finance":  "#bcbd22",  # olive/gold
}

# Canonical task display order (used as x-axis order, bar order within islands, etc.).
TASK_ORDER = ["advice", "summarization", "tutor", "critique"]
DOMAIN_ORDER = ["medical", "sports", "finance"]
