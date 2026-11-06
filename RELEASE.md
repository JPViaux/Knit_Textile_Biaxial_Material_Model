# Release procedure and permanent archival

The publication repository is https://github.com/JPViaux/Knit_Textile_Biaxial_Material_Model.
The first package version is **1.0.0**, tag **v1.0.0**. GitHub availability and a GitHub release do not by themselves provide a permanent archive DOI.

## Before publishing a version

1. Run `verify_package.py` in VSCode with checksum and scientific checks enabled, using the documented environment. Inspect `2_Code/VERIFICATION.md` for the tested platform and limits.
2. Confirm the paper metadata and update `CITATION.cff` if needed. The paper title/authors are confirmed; journal publication details and DOI remain pending.
3. Use `rebuild_manifest.py` only after reviewing intentional package changes. Commit the resulting manifest with those changes. Verification never silently rebuilds checksums.
4. Review `git status`: omit generated outputs, build binaries, caches, temporary dependencies and OneDrive conflict files. Keep both numbered folders together.
5. Commit the reviewed changes and verify that the intended tag points to that commit. The existing unpublished local v1.0.0 tag predates the readability cleanup; refresh it before upload if using that version. Then create a GitHub release named `Publication package v1.0.0` at tag `v1.0.0`, using `RELEASE_NOTES.md`. Do not replace a published tag when scientific inputs change; issue a new version.

## Licensing and Zenodo: pending author decision

No code or data license is assigned. The authors requested postponing this decision. Public GitHub access is possible, but reuse permissions are unresolved. Select appropriate code/data licenses and confirm ownership before enabling the permanent archive.

Once decided, connect the repository through Zenodo's GitHub integration, enable archival, then publish a new GitHub release. Verify the deposited files and its version DOI. Add the verified DOI to `CITATION.cff`, README and the manuscript data-availability statement. Do not invent a DOI or claim a deposit before Zenodo confirms it.

Suggested data-availability text **after public upload has been verified**:

> Experimental data, the constitutive model, calibration inputs and archived results are available at https://github.com/JPViaux/Knit_Textile_Biaxial_Material_Model (publication package v1.0.0).

The manuscript itself has not been edited by this packaging workflow. Add the archive DOI to that statement when one is actually issued.
