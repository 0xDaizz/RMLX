use clap::{Parser, Subcommand};

mod config;
mod hostfile;
mod launch;
mod ssh;
mod tb_discovery;
mod topology;

#[derive(Parser)]
#[command(name = "rmlx", about = "RMLX distributed tooling")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch distributed RMLX jobs across hosts via SSH
    Launch(launch::LaunchArgs),
    /// Generate hostfiles and configure hosts for distributed runs
    Config(config::ConfigArgs),
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.command {
        Commands::Launch(args) => launch::run(args),
        Commands::Config(args) => config::run(args),
    };
    std::process::exit(code);
}
