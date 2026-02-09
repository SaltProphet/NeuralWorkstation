#!/usr/bin/env python3
"""
FORGE v1 - Unified Neural Audio Workstation
    }
    
    /* Row spacing */
    .gr-row {
        margin: 10px 0 !important;
    }
    
/* Column spacing */
.gr-column {
    padding: 10px !important;
}
"""

def create_gradio_interface():
    """
    Create the main Gradio interface with FORGE Neural Workstation styling.
    Implements dark theme with orange accents, multi-phase tabs, and persistent console.
    """
                        demucs_btn.click(
                            fn=demucs_wrapper,
                            inputs=[demucs_audio, demucs_model, demucs_cache],
                            outputs=[demucs_output, demucs_status]
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1,
        # Footer
        gr.Markdown("---")
        gr.Markdown("<span style='color:#ff6600;'>FORGE v1 - Neural Audio Workstation</span> | <span style='color:#ffb366;'>Built by NeuralWorkstation Team</span>")
    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point for FORGE v1 application.
    Sets up directories and launches the Gradio interface.
    """
    
    print("=" * 70)
    print("FORGE v1 - Neural Audio Workstation")
    print("=" * 70)
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    print("✅ Directories ready")
    
    # Create and launch Gradio app
    print("\n🚀 Launching Gradio interface...")
    app = create_gradio_interface()
    
    # Launch with share=False for local use, share=True for public link
    # Disable SSR mode to fix "Could not get API info" errors in Gradio 5.x
    # In Gradio 6.x, theme and css are passed to launch() instead of Blocks()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        ssr_mode=False,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS
    )


if __name__ == "__main__":
    main()
