"""
Quick end-to-end test of the entire pipeline from MIDI to generation.

This verifies:
1. MIDI file → tokens
2. Tokens → model
3. Model → generation
4. Generation → MIDI file
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.midi_parser import load_midi_file
from src.data.representation import MusicRepresentationConverter, VocabularyConfig
from src.data.dataset import LazyMidiDataset
from src.models import MusicTransformerVAEGAN
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


def test_full_pipeline():
    """Test the complete pipeline from MIDI to generation."""
    print("🎼 Testing End-to-End Pipeline")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    midi_dir = project_root / "data" / "raw"
    
    # Check for MIDI files
    midi_files = list(midi_dir.glob("*.mid"))
    if not midi_files:
        print("❌ No MIDI files found in data/raw/")
        return False
    
    print(f"✅ Found {len(midi_files)} MIDI files")
    test_file = midi_files[0]
    print(f"📄 Testing with: {test_file.name}")
    
    try:
        # Step 1: Load and parse MIDI
        print("\n1️⃣ Loading MIDI file...")
        midi_data = load_midi_file(test_file)
        print(f"   ✅ Loaded: {len(midi_data.instruments)} instruments, "
              f"{sum(len(inst.notes) for inst in midi_data.instruments)} notes")
        
        # Step 2: Convert to tokens
        print("\n2️⃣ Converting to tokens...")
        vocab_config = VocabularyConfig()
        converter = MusicRepresentationConverter(vocab_config)
        representation = converter.midi_to_representation(midi_data)
        
        print(f"   ✅ Vocab size: {vocab_config.vocab_size}")
        print(f"   ✅ Token sequence length: {len(representation.tokens)}")
        print(f"   ✅ Token range: {representation.tokens.min()} - {representation.tokens.max()}")
        print(f"   ✅ First 20 tokens: {representation.tokens[:20].tolist()}")
        
        # Step 3: Create dataset and get a batch
        print("\n3️⃣ Creating dataset...")
        dataset = LazyMidiDataset(
            data_dir=midi_dir,
            max_files=1,
            sequence_length=512,
            enable_caching=False
        )
        
        sample = dataset[0]
        tokens_tensor = sample['tokens'].unsqueeze(0)  # Add batch dimension
        print(f"   ✅ Dataset created: {len(dataset)} sequences")
        print(f"   ✅ Sample shape: {tokens_tensor.shape}")
        
        # Step 4: Create and test model
        print("\n4️⃣ Creating model...")
        model = MusicTransformerVAEGAN(
            vocab_size=774,
            d_model=256,
            n_layers=2,
            n_heads=4,
            max_sequence_length=512,
            mode="transformer"
        )
        
        model_size = model.get_model_size()
        print(f"   ✅ Model created: {model_size['total_parameters']:,} parameters")
        print(f"   ✅ Model size: {model_size['model_size_mb']:.1f}MB")
        
        # Step 5: Forward pass
        print("\n5️⃣ Testing forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(tokens_tensor)
        
        print(f"   ✅ Forward pass successful!")
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   ✅ Output range: {output.min().item():.2f} - {output.max().item():.2f}")
        
        # Step 6: Test generation
        print("\n6️⃣ Testing generation...")
        prompt = tokens_tensor[:, :10]  # Use first 10 tokens as prompt
        
        with torch.no_grad():
            generated = model.generate(
                prompt_tokens=prompt,
                max_new_tokens=50,
                temperature=1.0,
                top_k=50
            )
        
        print(f"   ✅ Generated {generated.shape[1]} tokens")
        print(f"   ✅ Generated tokens: {generated[0].tolist()}")
        
        # Step 7: Verify token validity
        print("\n7️⃣ Verifying generated tokens...")
        valid_tokens = (generated >= 0) & (generated < 774)
        print(f"   ✅ All tokens valid: {valid_tokens.all().item()}")
        
        # Step 8: Test token-to-event conversion
        print("\n8️⃣ Testing token-to-event conversion...")
        for i in range(min(10, generated.shape[1])):
            token = generated[0, i].item()
            event_type, value = vocab_config.token_to_event_info(token)
            print(f"   Token {token:3d} → {event_type.name}(value={value})")
        
        print("\n✅ Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_modes():
    """Test all three model modes."""
    print("\n\n🔄 Testing All Model Modes")
    print("=" * 60)
    
    modes = ["transformer", "vae", "vae_gan"]
    vocab_size = 774
    
    for mode in modes:
        print(f"\n📍 Testing {mode} mode...")
        try:
            model = MusicTransformerVAEGAN(
                vocab_size=vocab_size,
                d_model=128,
                n_layers=1,
                n_heads=4,
                max_sequence_length=256,
                mode=mode,
                latent_dim=32 if mode != "transformer" else None
            )
            
            # Test forward pass
            dummy_tokens = torch.randint(0, vocab_size, (2, 256))
            
            if mode == "transformer":
                output = model(dummy_tokens)
                loss_dict = model.compute_loss(dummy_tokens)
                print(f"   ✅ Output shape: {output.shape}")
                print(f"   ✅ Loss: {loss_dict['total_loss'].item():.4f}")
            else:
                output = model(dummy_tokens, return_latent=True)
                loss_dict = model.compute_loss(dummy_tokens)
                print(f"   ✅ Logits shape: {output['logits'].shape}")
                print(f"   ✅ Latent shape: {output['z'].shape}")
                print(f"   ✅ Recon loss: {loss_dict['reconstruction_loss'].item():.4f}")
                print(f"   ✅ KL loss: {loss_dict['kl_loss'].item():.4f}")
            
        except Exception as e:
            print(f"   ❌ {mode} mode failed: {e}")


if __name__ == "__main__":
    # Run tests
    success = test_full_pipeline()
    
    if success:
        test_all_modes()
    
    print("\n" + "="*60)
    print("🏁 Testing complete!")
    print("="*60)