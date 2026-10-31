# Personal end-to-end workflow

Everything below keeps private data local until the one explicit Kaggle upload.
`.secrets/`, `data/whatsapp_export/` and `data/personal/` are git-ignored.

1. **Export chats** (text only) from a decrypted `msgstore.db`:
   `python scripts/export_whatsapp_db.py --db <msgstore.db> --me "<Your Name>" --adb-contacts`
2. **Build your one-take script** from your own sentences (PII, names and family words filtered):
   `python scripts/build_personal_script.py`
3. **Record it once** in the studio (`python -m echotwin.ui`, Voice tab). Long takes are
   normalised and split at pauses into 6-25 s clips with script-aligned transcripts
   (`src/echotwin/voice/segmentation.py`); transcripts are labelled *unverified*.
4. **Grant consent** (your own voice only) in the Voice tab, then *Select references*.
5. **Sanitise and upload privately**:
   `python scripts/prepare_private_kaggle_data.py` then
   `kaggle datasets create -p data/personal/kaggle --dir-mode zip`
   (other people are renamed "Person A/B/...", phone numbers/emails/URLs redacted; messages
   may still contain names typed inside text.)
6. **Train + clone on a T4**:
   `python scripts/build_kaggle_notebook.py --accelerator personal` then
   `kaggle kernels push -p notebooks/kaggle`. The kernel trains the LoRA adapter, evaluates
   all methods on the same held-out prompts, and runs VoxCPM synthesis only if
   `voice_consent.json` is present in the dataset.
