#!/bin/zsh

# Claude Code Proxy Installation Script
# This script installs the Claude proxy globally for easy access

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_NAME="claude-code-proxy"
SCRIPT_NAME="claude-proxy"

# Get the current project directory
PROJECT_DIR="$(cd "$(dirname "${0}")" && pwd)"
BIN_DIR="$PROJECT_DIR/bin"
SCRIPT_PATH="$BIN_DIR/$SCRIPT_NAME"

# Detect the shell and get appropriate rc file
get_shell_rc() {
    local shell_name=$(basename "$SHELL")
    
    case "$shell_name" in
        zsh)
            echo "$HOME/.zshrc"
            ;;
        bash)
            if [[ -f "$HOME/.bash_profile" ]]; then
                echo "$HOME/.bash_profile"
            else
                echo "$HOME/.bashrc"
            fi
            ;;
        *)
            echo "$HOME/.profile"
            ;;
    esac
}

# Check if the script can be run
precheck() {
    echo -e "${BLUE}🔍 Checking system compatibility...${NC}"
    
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo -e "${RED}Error: Main script file not found at $SCRIPT_PATH${NC}"
        exit 1
    fi
    
    if [[ ! -x "$SCRIPT_PATH" ]]; then
        chmod +x "$SCRIPT_PATH"
        echo -e "${GREEN}✅ Set script execution permissions${NC}"
    fi
    
    echo -e "${GREEN}✅ Pre-check passed${NC}"
}

# Find a directory in PATH where we can write
create_symlink_path() {
    local suggested_dirs=("$HOME/.local/bin" "$HOME/bin" "/usr/local/bin")
    local target_dir=""
    
    # Check if we have permissions in suggested directories
    for dir in "$@"; do
        if [[ -d "$dir" ]] && [[ -w "$dir" ]]; then
            target_dir="$dir"
            break
        fi
    done
    
    # If no suitable directory found, create ~/.local/bin
    if [[ -z "$target_dir" ]]; then
        target_dir="$HOME/.local/bin"
        if [[ ! -d "$target_dir" ]]; then
            mkdir -p "$target_dir"
            echo -e "${YELLOW}Created directory: $target_dir${NC}"
        fi
    fi
    
    echo "$target_dir"
}

# Add directory to PATH if not already in PATH
add_to_path() {
    local target_dir=$1
    local rc_file=$2
    
    if [[ ":$PATH:" != *":$target_dir:"* ]]; then
        echo "" >> "$rc_file"
        echo "# Claude Code Proxy - Add $target_dir to PATH" >> "$rc_file"
        echo "export PATH=\"$target_dir:\$PATH\"" >> "$rc_file"
        echo -e "${GREEN}Added $target_dir to PATH (written to $rc_file)${NC}"
        return 0
    fi
    
    return 1
}

# Install the script
install_script() {
    echo -e "${BLUE}📦 Installing Claude Code Proxy...${NC}"
    
    local target_dir=$(create_symlink_path)
    local symlink_path="$target_dir/$SCRIPT_NAME"
    
    # Check if symlink already exists
    if [[ -L "$symlink_path" ]] || [[ -f "$symlink_path" ]]; then
        echo -e "${YELLOW}⚠️  $symlink_path already exists and will be replaced${NC}"
        rm -f "$symlink_path"
    fi
    
    # Create symlink
    ln -sf "$SCRIPT_PATH" "$symlink_path"
    echo -e "${GREEN}✅ Created symbolic link: $symlink_path -> $SCRIPT_PATH${NC}"
    
    # Add to PATH if necessary
    local rc_file=$(get_shell_rc)
    if add_to_path "$target_dir" "$rc_file"; then
        echo -e "${YELLOW}🔔 Please add $target_dir to PATH or reload shell${NC}"
        echo -e "${YELLOW}   You can run: source $rc_file${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 Installation complete!${NC}"
    echo ""
    echo -e "${BLUE}Usage instructions:${NC}"
    echo "  • Start server: $SCRIPT_NAME"
    echo "  • Development mode: $SCRIPT_NAME -d"
    echo "  • Custom port: $SCRIPT_NAME -p 8080"
    echo "  • Docker launch: $SCRIPT_NAME --docker"
    echo "  • View help: $SCRIPT_NAME --help"
    echo ""
    echo -e "${YELLOW}Configuration tips:${NC}"
    echo "  1. Edit .env file in project directory to configure server settings"
    echo "  2. Modify models.yaml for model settings"
    echo "  3. Project directory: $PROJECT_DIR"
}

# Main installation process
main() {
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          Claude Code Proxy Installer             ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
    
    precheck
    install_script
    
    echo -e "${GREEN}You can now use the $SCRIPT_NAME command from any directory!${NC}"
}

# Handle special cases
if [[ "$1" == "--check" ]]; then
    if [[ -L "/usr/local/bin/$SCRIPT_NAME" ]] || [[ -f "$HOME/.local/bin/$SCRIPT_NAME" ]] || [[ -f "$HOME/bin/$SCRIPT_NAME" ]]; then
        echo -e "${GREEN}已安装${NC}"
        exit 0
    else
        echo -e "${RED}未安装${NC}"
        exit 1
    fi
fi

# Run main installation
main "$@"
