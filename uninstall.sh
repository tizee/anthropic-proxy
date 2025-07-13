#!/bin/zsh

# Claude Code Proxy Uninstallation Script
# This script removes the globally installed Claude proxy

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_NAME="claude-code-proxy"
SCRIPT_NAME="claude-proxy"

# Function to remove symlink and clean up
remove_symlink() {
    local target_dir=$1
    local symlink_path="$target_dir/$SCRIPT_NAME"
    
    if [[ -L "$symlink_path" ]]; then
        rm -f "$symlink_path"
        echo -e "${GREEN}✅ Removed symbolic link: $symlink_path${NC}"
        return 0
    elif [[ -f "$symlink_path" ]]; then
        echo -e "${YELLOW}⚠️  Warning: $symlink_path is a regular file, skipping removal${NC}"
        return 1
    fi
    
    return 1
}

# Remove from PATH in shell configuration
remove_from_path() {
    local target_dir=$1
    local rc_file=$2
    
    # Create backup and remove PATH modification
    cp "$rc_file" "$rc_file.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Remove the PATH addition
    sed -i.bak "/Claude Code Proxy - Add $target_dir to PATH/,+1d" "$rc_file" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  PATH configuration not found or unable to modify $rc_file${NC}"
    }
}

# Main uninstallation process
main() {
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         Claude Code Proxy Uninstaller            ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
    
    local search_dirs=()
    
    # Check common directories for the script
    if [[ -d "$HOME/.local/bin" ]]; then
        search_dirs+=("$HOME/.local/bin")
    fi
    
    if [[ -d "$HOME/bin" ]]; then
        search_dirs+=("$HOME/bin")
    fi
    
    if [[ -d "/usr/local/bin" ]] && [[ -w "/usr/local/bin" ]]; then
        search_dirs+=("/usr/local/bin")
    fi
    
    # Also check /usr/bin and /bin
    search_dirs+=("/usr/bin" "/bin")
    
    local removed_count=0
    
    echo -e "${BLUE}🔍 Searching for installed files...${NC}"
    
    for dir in "${search_dirs[@]}"; do
        local symlink_path="$dir/$SCRIPT_NAME"
        if [[ -L "$symlink_path" ]]; then
            remove_symlink "$dir"
            ((removed_count++))
            
            # Detect shell and clean up PATH configuration
            local shell_name=$(basename "$SHELL")
            local rc_file=""
            
            case "$shell_name" in
                zsh)
                    rc_file="$HOME/.zshrc"
                    ;;
                bash)
                    if [[ -f "$HOME/.bash_profile" ]]; then
                        rc_file="$HOME/.bash_profile"
                    else
                        rc_file="$HOME/.bashrc"
                    fi
                    ;;
                *)
                    rc_file="$HOME/.profile"
                    ;;
            esac
            
            if [[ -f "$rc_file" ]]; then
                remove_from_path "$dir" "$rc_file"
            fi
        fi
    done
    
    if [[ $removed_count -eq 0 ]]; then
        echo -e "${YELLOW}⚠️  No installed $SCRIPT_NAME found${NC}"
        echo -e "${YELLOW}   Possible reasons: installation failed, manually deleted, or path not in standard locations${NC}"
    else
        echo ""
        echo -e "${GREEN}🎉 Uninstallation complete!${NC}"
        echo -e "${YELLOW}Note: After uninstallation, you may need to reload your shell configuration:${NC}"
        echo "  source ~/.zshrc  # or your corresponding shell configuration file"
    fi
}

# Ask for confirmation if no arguments
if [[ "$1" != "--force" ]]; then
    echo -e "${YELLOW}This will remove the globally installed claude-proxy command.${NC}"
    echo -e "${YELLOW}Do you want to continue? [y/N]:${NC} \c"
    read -r response
    
    case $response in
        [yY][eE][sS]|[yY])
            echo -e "${GREEN}Starting uninstallation...${NC}"
            ;;
        *)
            echo -e "${YELLOW}Uninstallation cancelled.${NC}"
            exit 0
            ;;
    esac
fi

# Run main uninstallation
main "$@"