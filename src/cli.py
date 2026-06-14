"""CLI entry point for the equity analysis pipeline."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table
from rich import box

from .config import load_config, load_dotenv, available_sectors
from .pipeline import run, scan_sector, TickerResult, ScanResult

console = Console()


def main() -> None:
    load_dotenv()  # pull GOOGLE_AI_STUDIO_API_KEY etc. from .env if present
    parser = argparse.ArgumentParser(
        description="Equity Analysis Pipeline — research aid, not investment advice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli                           # run with default config
  python -m src.cli --tickers AAPL MSFT       # override tickers
  python -m src.cli --config config/my.yaml   # custom config file
  python -m src.cli --no-llm                  # skip LLM calls (scores only)
  python -m src.cli --scan-sector --sector technology   # wide screen + deep dive on flagged names
        """,
    )
    parser.add_argument(
        "--config", default="config/run_config.yaml",
        help="Path to run config YAML (default: config/run_config.yaml)",
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Override tickers from config",
    )
    parser.add_argument(
        "--sector",
        help="Sector peer universe to rank against (required, unless set in the config). "
             "Must match a file in config/peers/, e.g. technology, financials, energy.",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable all LLM calls (computed scores only)",
    )
    parser.add_argument(
        "--llm-provider", metavar="NAME",
        help="Prioritize a configured LLM provider by name (e.g. ollama, "
             "google_ai_studio, openrouter). It's moved to the front of the "
             "fallback chain; the others remain as fallbacks. Ignored with --no-llm.",
    )
    parser.add_argument(
        "--output-dir", default="reports",
        help="Directory for markdown reports (default: reports/)",
    )
    parser.add_argument(
        "--no-peer-refresh", action="store_true",
        help="Skip peer universe refresh (use cached distributions)",
    )
    parser.add_argument(
        "--scan-sector", action="store_true",
        help="Idea-generation mode: score the whole sector peer universe (no LLM), "
             "flag the most value-divergent names, then run the full LLM analysis "
             "only on those. Ignores --tickers. Emits one combined report.",
    )
    parser.add_argument(
        "--deep-count", type=int, default=3, metavar="N",
        help="Number of flagged names to analyze deeply in --scan-sector (default: 3)",
    )

    args = parser.parse_args()

    # Load config
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    # Apply CLI overrides
    if args.tickers:
        cfg.tickers = [t.upper() for t in args.tickers]
    if args.sector:
        cfg.sector = args.sector

    # Sector is mandatory and must have a peer universe — scoring against the wrong
    # peer set (or none) produces misleading percentiles, so fail loudly instead.
    sectors = available_sectors()
    if not cfg.sector:
        console.print(
            "[red]No sector specified.[/red] Pass [bold]--sector NAME[/bold] or set "
            "[bold]sector:[/bold] in the config.\n"
            f"  Available sectors: {', '.join(sectors) or '(none found in config/peers/)'}"
        )
        sys.exit(1)
    if cfg.sector not in sectors:
        console.print(
            f"[red]Unknown sector '{cfg.sector}'.[/red] No peer file at "
            f"config/peers/{cfg.sector}.yaml.\n"
            f"  Available sectors: {', '.join(sectors) or '(none found in config/peers/)'}"
        )
        sys.exit(1)
    if args.no_llm:
        cfg.llm_providers = []  # disable all providers
    elif args.llm_provider:
        names = [p.name for p in cfg.llm_providers]
        if args.llm_provider not in names:
            console.print(
                f"[red]Unknown --llm-provider '{args.llm_provider}'. "
                f"Configured providers: {', '.join(names) or 'none'}[/red]"
            )
            sys.exit(1)
        # Move the chosen provider to the front; keep the rest as fallbacks.
        cfg.llm_providers = (
            [p for p in cfg.llm_providers if p.name == args.llm_provider]
            + [p for p in cfg.llm_providers if p.name != args.llm_provider]
        )

    # Print run header
    console.print()
    console.rule("[bold blue]Equity Analysis Pipeline[/bold blue]")
    if args.scan_sector:
        console.print(f"  Mode    : sector scan (wide → deep, top {args.deep_count})")
        console.print(f"  Sector  : {cfg.sector}")
    else:
        console.print(f"  Tickers : {', '.join(cfg.tickers)}")
        console.print(f"  Sector  : {cfg.sector}")
    provider = cfg.first_available_provider()
    if provider:
        console.print(f"  LLM     : {provider.name} / {provider.model}")
    else:
        console.print("  LLM     : [yellow]no API key — computed scores only[/yellow]")
    console.print()

    # Scan mode: wide screen + deep dive on flagged names, then exit.
    if args.scan_sector:
        try:
            scan = scan_sector(cfg, output_dir=args.output_dir,
                               deep_count=args.deep_count)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Scan error: {e}[/red]")
            raise
        _print_scan_summary(scan)
        return

    # Run pipeline
    try:
        result = run(cfg, output_dir=args.output_dir)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Pipeline error: {e}[/red]")
        raise

    # Print scorecard table
    console.print()
    console.rule("[bold blue]Factor Scorecard[/bold blue]")
    _print_scorecard(result.ticker_results)

    # Print report paths
    console.print()
    console.rule("[bold blue]Reports[/bold blue]")
    for r in result.ticker_results:
        if r.report_path:
            console.print(f"  {r.ticker}: [cyan]{r.report_path}[/cyan]")

    # Print comparative ranking if available
    if result.comparative:
        ranking = result.comparative.get("overall_ranking", [])
        if ranking:
            console.print()
            console.rule("[bold blue]Comparative Ranking[/bold blue]")
            console.print(f"  {' > '.join(ranking)}")
            console.print(f"  {result.comparative.get('ranking_rationale', '')}")

    console.print()
    console.print(f"[dim]Run ID: {result.run_id} | DB: {cfg.db_path}[/dim]")
    console.print()


def _print_scan_summary(scan: ScanResult) -> None:
    # Ranked sector screen
    console.print()
    console.rule("[bold blue]Sector Screen[/bold blue]")
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("", width=2)
    table.add_column("Ticker", style="bold", width=8)
    for col in ("Val", "Qual", "Grow", "Mom", "Insdr", "Comp"):
        table.add_column(col, justify="center", width=6)

    ranked = sorted(
        scan.wide_results,
        key=lambda r: (r.scores.get("composite") or 0.0), reverse=True,
    )
    for r in ranked:
        flag = "[magenta]★[/magenta]" if r.ticker in scan.qualifying else ""
        s = r.scores

        def _c(key: str) -> str:
            v = s.get(key)
            if v is None:
                return "[dim]—[/dim]"
            return _score_cell(float(v), key == "composite")

        table.add_row(
            flag, r.ticker, _c("valuation"), _c("quality"), _c("growth"),
            _c("momentum"), _c("insider"), _c("composite"),
        )
    console.print(table)
    console.print("[dim]★ = value-divergence flag (high quality+growth, low valuation)[/dim]")

    # Flagged / deep
    console.print()
    console.rule("[bold blue]Deep Analysis[/bold blue]")
    if scan.flagged:
        console.print(f"  Analyzed: [cyan]{', '.join(scan.flagged)}[/cyan]")
        for r in scan.deep_results:
            if r.synthesis:
                lean = r.synthesis.get("buy_sell_lean", "N/A")
                conf = r.synthesis.get("lean_confidence", "")
                lc = _lean_color(lean)
                console.print(f"    {r.ticker}: [{lc}]{lean}[/{lc}] ({conf} confidence)")
    else:
        console.print("  [yellow]No names flagged for deep analysis.[/yellow]")

    if scan.comparative:
        ranking = scan.comparative.get("overall_ranking", [])
        if ranking:
            console.print(f"  Ranking: {' > '.join(ranking)}")

    console.print()
    console.rule("[bold blue]Report[/bold blue]")
    console.print(f"  [cyan]{scan.scan_report_path}[/cyan]")
    console.print()
    console.print(f"[dim]Run ID: {scan.run_id}[/dim]")
    console.print()


def _print_scorecard(ticker_results: list[TickerResult]) -> None:
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Factor", style="bold", width=18)

    for r in ticker_results:
        table.add_column(r.ticker, justify="center", width=10)

    factors = [
        ("Valuation", "valuation"),
        ("Quality", "quality"),
        ("Growth", "growth"),
        ("Momentum", "momentum"),
        ("Insider", "insider"),
        ("Filing Tone *", "filing_tone"),
        ("News Tone *", "news_tone"),
        ("─" * 14, None),
        ("COMPOSITE", "composite"),
    ]

    for label, key in factors:
        if key is None:
            table.add_row(label, *["" for _ in ticker_results])
            continue
        row = [label]
        for r in ticker_results:
            val = r.scores.get(key)
            if val is None:
                row.append("[dim]N/A[/dim]")
            else:
                row.append(_score_cell(float(val), key == "composite"))
        table.add_row(*row)

    console.print(table)
    console.print("[dim]* LLM-judged scores (qualitative estimate)[/dim]")

    # Buy/sell leans
    has_lean = any(r.synthesis for r in ticker_results)
    if has_lean:
        console.print()
        for r in ticker_results:
            if r.synthesis:
                lean = r.synthesis.get("buy_sell_lean", "N/A")
                conf = r.synthesis.get("lean_confidence", "")
                lean_color = _lean_color(lean)
                console.print(
                    f"  {r.ticker}: [{lean_color}]{lean}[/{lean_color}] "
                    f"({conf} confidence)"
                )


def _score_cell(val: float, is_composite: bool) -> str:
    if is_composite:
        if val >= 4.0:
            return f"[green]{val:.2f}[/green]"
        elif val >= 3.0:
            return f"[yellow]{val:.2f}[/yellow]"
        else:
            return f"[red]{val:.2f}[/red]"
    else:
        score = round(val)
        if score >= 4:
            return f"[green]{val:.1f}[/green]"
        elif score == 3:
            return f"[yellow]{val:.1f}[/yellow]"
        else:
            return f"[red]{val:.1f}[/red]"


def _lean_color(lean: str) -> str:
    if lean in ("strong_buy", "buy"):
        return "green"
    elif lean in ("strong_sell", "sell"):
        return "red"
    return "yellow"


if __name__ == "__main__":
    main()
