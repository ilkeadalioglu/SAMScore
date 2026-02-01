# Changelog

All notable changes, known issues, and pending fixes for this project.

---

### ✅ Recently Completed Features

#### SAM Loss Integration with Dynamic Scheduling
- Added dynamic SAM loss scheduling with multiple policies (linear, exponential, cosine, step)
- Implemented `--sam_start_epoch`, `--sam_warmup_epochs`, `--sam_schedule_policy` options
- Added SAM weight visualization and logging
- See [DYNAMIC_SAM_SCHEDULING.md](documentation/DYNAMIC_SAM_SCHEDULING.md)

#### Video Generation Module
- Implemented temporal blending algorithm for smooth video generation
- Added `generate_videos.py` CLI script
- Integrated with test.py via `--generate_videos` flag
- Support for multiple codecs (FFV1, mp4v, avc1)
- See [VIDEO_GENERATION_README.md](documentation/VIDEO_GENERATION_README.md)

#### Multi-Epoch Testing
- Added `--experiment_path` option for simplified testing
- Support for comma-separated epoch lists: `--epochs 50,100,150,200`
- Automatic video generation after testing (optional)
- Frame-based output naming: `val_e{epoch}_p{patient}_f{frame}.png`

#### Self-ONN Architecture Support
- Implemented Self-ONN UNet generator
- Added Self-ONN discriminator variants
- Guided Attention Blocks for encoder-decoder connections
- See [SELFONN_UNET_IMPLEMENTATION.md](documentation/SELFONN_UNET_IMPLEMENTATION.md)

#### MTSNet Generator
- Multi-scale Transformer-based generator architecture
- Configurable attention windows and scales
- See [mts_blocks.py](models/mts_blocks.py) and [example_usage.py](example_usage.py)

---

## Version History

### [0.9.0] - Current State (Pre-Release)

#### Added
- Comprehensive README.md with full documentation
- SAM loss scheduling functionality
- Video generation module
- Multi-epoch testing support
- Self-ONN and MTSNet architectures
- Loss visualization tools
- Learning rate visualization
- Validation monitoring
- SLURM/HPC support scripts

#### Changed
- Refactored network factory for better modularity
- Improved dataset handling for validation
- Enhanced options parsing with experiment_path support
- Updated documentation structure

#### Fixed
- SAM loss calculation issues
- Resolution handling in generators
- Video generation temporal blending
- Checkpoint loading for specific epochs

#### Known Issues
- **test.py docstring corruption** (lines 2-25) - TO BE FIXED

---

## Pre-Release Testing Checklist

Before pushing to GitHub, verify:

- [ ] Fix test.py docstring (remove lines 2-25, restore proper documentation)
- [ ] Test training with CycleGAN baseline
- [ ] Test training with SAM loss
- [ ] Test multi-epoch inference
- [ ] Test video generation
- [ ] Verify all documentation links work
- [ ] Check requirements.txt is complete
- [ ] Ensure .gitignore excludes checkpoints and large files
- [ ] Test on clean environment
- [ ] Verify HPC scripts work
- [ ] Run linter/formatter

---

## Roadmap

### Planned Features

#### Short Term
- [ ] Fix test.py docstring bug
- [ ] Add automated testing suite
- [ ] Improve error handling and logging
- [ ] Add progress bars for long operations
- [ ] Implement checkpoint cleanup utility

#### Medium Term
- [ ] Add FID and other metrics calculation
- [ ] Implement distributed training support
- [ ] Add TensorBoard integration
- [ ] Create interactive Jupyter notebooks
- [ ] Add model ensemble support

#### Long Term
- [ ] Web interface for model training/testing
- [ ] Pre-trained model zoo
- [ ] Automatic hyperparameter tuning
- [ ] Integration with medical imaging viewers
- [ ] Real-time inference API

---

## Migration Notes

### From Original CUT/CycleGAN

If migrating from the original CUT/CycleGAN repository:

1. **New Options**:
   - `--lambda_seg_consistency`: Enable SAM loss
   - `--sam_start_epoch`: Start SAM at specific epoch
   - `--sam_warmup_epochs`: Gradual SAM weight increase
   - `--sam_schedule_policy`: Scheduling policy
   - `--experiment_path`: Simplified testing path
   - `--epochs`: Multi-epoch testing
   - `--generate_videos`: Automatic video generation

2. **New Architectures**:
   - Self-ONN generators and discriminators
   - MTSNet generator
   - Attention-enhanced ResNet variants

3. **New Scripts**:
   - `generate_videos.py`: Video generation
   - `test_with_video.py`: Testing with video output
   - `example_usage.py`: Architecture examples

4. **Checkpoint Structure**:
   - Checkpoints now in `nets/` subfolder
   - Validation images in `val/` subfolder
   - Videos in `videos/` subfolder

---

## Breaking Changes

### None (Initial Release)

All features are backwards compatible with original CUT/CycleGAN usage.

---

## Deprecation Notices

### None Currently

---

## Bug Fixes History

### Critical Fixes
- **SAM Loss Calculation**: Fixed tensor shape mismatches in SAM consistency loss
- **Resolution Issues**: Fixed generator output resolution inconsistencies
- **Video Writer**: Fixed codec compatibility issues across platforms

### Minor Fixes
- Improved error messages for missing checkpoints
- Fixed learning rate scheduling edge cases
- Corrected validation image naming consistency
- Fixed memory leaks in long training runs

---

## Documentation Updates

### New Documentation Files
- `README.md` - Main project documentation
- `CHANGELOG.md` - This file
- `CONTRIBUTING.md` - Contribution guidelines
- `documentation/QUICK_START_VIDEO.md` - Video generation quick start
- `documentation/DYNAMIC_SAM_QUICK_START.md` - SAM scheduling quick start

### Updated Documentation
- `documentation/README.md` - Original CUT documentation
- All implementation summaries updated with current state

---

## Contributors

- Main Development: [Your Name]
- Original CUT/CycleGAN: Taesung Park, Jun-Yan Zhu, et al.
- SAM Integration: Based on Meta AI's Segment Anything Model
- MedSAM: Based on bowang-lab's MedSAM

---

## Release Notes Format

Future releases will follow this format:

```
## [Version] - YYYY-MM-DD

### Added
- New features

### Changed  
- Changes to existing functionality

### Deprecated
- Features to be removed

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security fixes
```

---

**Last Updated**: 2025-12-11



