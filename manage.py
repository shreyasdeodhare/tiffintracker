import click

from app import create_app
from app.application import accounts

app = create_app()


@app.cli.command("reconcile-week")
@click.option("--week", help="ISO date within the week; defaults to current business week")
def reconcile_week_command(week):
    """Read-only independent financial checks, emitting no member names/emails."""
    import json
    from datetime import date

    from app.reconciliation import reconcile_week
    from app.services import local_today, week_bounds

    try:
        day = date.fromisoformat(week) if week else local_today()
    except ValueError as exc:
        raise click.BadParameter("Use YYYY-MM-DD", param_hint="--week") from exc
    result = reconcile_week(*week_bounds(day))
    click.echo(json.dumps(result, indent=2))
    if not result["financially_consistent"]:
        raise click.ClickException("Financial discrepancy detected; no data changed")


def run_account(operation, *args):
    from sqlalchemy.exc import IntegrityError

    from app.domain.orders import BusinessError

    try:
        return operation(*args)
    except (BusinessError, IntegrityError) as exc:
        raise click.ClickException(
            str(exc)
            if isinstance(exc, BusinessError)
            else "Account data conflicts with an existing record"
        ) from None


@app.cli.command("create-admin")
@click.option("--email", prompt=True)
@click.option("--name", prompt=True)
@click.password_option()
def create_admin(email, name, password):
    run_account(accounts.create_admin, name.strip(), email.strip().lower(), password)
    click.echo("Administrator created")


@app.cli.command("create-member")
@click.option("--email", prompt=True)
@click.option("--name", prompt=True)
@click.password_option()
def create_member(email, name, password):
    run_account(accounts.create_member, name.strip(), email.strip().lower(), password)
    click.echo("Member created; temporary password must be changed")


@app.cli.command("deactivate-member")
@click.argument("email")
def deactivate_member(email):
    run_account(accounts.activate_member, email.strip().lower(), False)
    click.echo("Member deactivated; history retained")


@app.cli.command("activate-member")
@click.argument("email")
def activate_member(email):
    run_account(accounts.activate_member, email.strip().lower(), True)
    click.echo("Member activated")


@app.cli.command("reset-member-password")
@click.argument("email")
@click.password_option("--temporary-password", confirmation_prompt=True)
def reset_member_password(email, temporary_password):
    run_account(accounts.reset_member_password, email.strip().lower(), temporary_password)
    click.echo("Password updated; communicate securely")


@app.cli.command("prune-login-attempts")
def prune_login_attempts():
    count = run_account(accounts.prune_login_attempts)
    click.echo(f"Removed {count} old login attempts")
