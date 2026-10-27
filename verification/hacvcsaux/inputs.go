package hacvcsaux

import (
	"bufio"
	"fmt"
	"os"
)

func PrepareInputs(path string) (err error) {
	m, err := ReadMatrix()
	if err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := file.Close(); err == nil {
			err = closeErr
		}
	}()
	out := bufio.NewWriter(file)
	if _, err = fmt.Fprintf(out, "hac-vcs-aux-inputs-v1 %s %d\n", MatrixSHA256, len(m.Cases)); err != nil {
		return err
	}
	for _, fixture := range m.Cases {
		aux, e := fixture.Aux(m.InputStride)
		if e != nil {
			return e
		}
		sizes := aux.SizesOut()
		if _, err = fmt.Fprintf(out, "%s %d %d %d", fixture.ID, fixture.BatchSize, fixture.FullChunkSizeOut, len(sizes)); err != nil {
			return err
		}
		for _, size := range sizes {
			if _, err = fmt.Fprintf(out, " %d", size); err != nil {
				return err
			}
		}
		if _, err = fmt.Fprintln(out); err != nil {
			return err
		}
	}
	return out.Flush()
}
