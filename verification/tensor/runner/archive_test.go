package runner

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"
)

func TestArchiveSymlinkAndTargetPins(t *testing.T) {
	dir := t.TempDir()
	root := filepath.Join(dir, "libtorch")
	os.MkdirAll(filepath.Join(root, "lib"), 0755)
	target := filepath.Join(root, "lib", "real.so")
	os.WriteFile(target, []byte("library bytes"), 0600)
	link := filepath.Join(root, "lib", "alias.so")
	os.Symlink("real.so", link)
	want, err := SHA(target)
	if err != nil {
		t.Fatal(err)
	}
	zpath := filepath.Join(dir, "archive.zip")
	file, err := os.Create(zpath)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(file)
	add := func(name, raw string, mode os.FileMode) {
		h := &zip.FileHeader{Name: name}
		h.SetMode(mode)
		f, err := w.CreateHeader(h)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = f.Write([]byte(raw)); err != nil {
			t.Fatal(err)
		}
	}
	add("libtorch/lib/alias.so", "real.so", os.ModeSymlink|0777)
	add("libtorch/lib/real.so", "library bytes", 0644)
	add("libtorch/lib/escape.so", "../../outside", os.ModeSymlink|0777)
	add("libtorch/lib/cycle.so", "cycle.so", os.ModeSymlink|0777)
	w.Close()
	file.Close()
	z, err := zip.OpenReader(zpath)
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	m, err := ArchiveMember(z, root, link, want)
	if err != nil || len(m) != 2 {
		t.Fatal(m, err)
	}
	os.WriteFile(target, []byte("changed"), 0600)
	if _, err = ArchiveMember(z, root, link, want); err == nil {
		t.Fatal("changed target accepted")
	}
	os.WriteFile(target, []byte("library bytes"), 0600)
	os.Remove(link)
	os.Symlink("different.so", link)
	if _, err = ArchiveMember(z, root, link, want); err == nil {
		t.Fatal("mismatched link text accepted")
	}
	os.Remove(link)
	os.WriteFile(link, []byte("library bytes"), 0600)
	if _, err = ArchiveMember(z, root, link, want); err == nil {
		t.Fatal("regular file substituted for link")
	}
	for _, name := range []string{"escape.so", "cycle.so"} {
		path := filepath.Join(root, "lib", name)
		dest := "../../outside"
		if name == "cycle.so" {
			dest = name
		}
		os.Symlink(dest, path)
		if _, err = ArchiveMember(z, root, path, want); err == nil {
			t.Fatal("escape or cycle accepted")
		}
	}
}
