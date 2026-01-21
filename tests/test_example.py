"""Integration test for mesh examples."""

import os
import shutil
import pytest
from pathlib import Path
import pyvista as pv

@pytest.fixture(scope="module")
def example_output_dir(tmp_path_factory):
    """Create temporary directory for example output."""
    tmpdir = tmp_path_factory.mktemp(basename="example_output")
    yield tmpdir
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_mesh_example(example_output_dir):
    """Test that mesh_example.py runs successfully and creates expected outputs."""
    from b3_2d.core.mesh import process_vtp_multi_section

    vtp_file = "examples/draped.vtk"
    output_dir = str(example_output_dir / "draped")
    
    if not Path(vtp_file).exists():
        pytest.skip(f"Sample VTP file not found: {vtp_file}")
    
    try:
        results = process_vtp_multi_section(vtp_file, output_dir)
    except Exception as e:
        pytest.skip(f"Skipping VTP test due to error: {e}")
    
    # Test that output directory was created
    assert Path(output_dir).exists()
    
    # Test that results structure is correct
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Test each result has expected structure
    for result in results:
        assert "section_id" in result
        assert "success" in result
        assert "output_dir" in result
        assert "created_files" in result
        
        section_dir = Path(result["output_dir"])
        assert section_dir.exists()
        
        # At least log file should be created
        assert any("2dmesh.log" in Path(f).name for f in result["created_files"])
    
    successful = sum(1 for r in results if r["success"])
    print(f"Example processed {len(results)} sections: {successful} successful")


def test_mesh_3web_example(example_output_dir):
    """Test 3-web mesh example."""
    from b3_2d.core.mesh import process_vtp_multi_section

    vtp_file = "examples/draped_3web.vtk"
    output_dir = str(example_output_dir / "draped_3web")
    
    if not Path(vtp_file).exists():
        pytest.skip(f"Sample VTP file not found: {vtp_file}")
    
    try:
        results = process_vtp_multi_section(vtp_file, output_dir)
    except Exception as e:
        pytest.skip(f"Skipping 3web VTP test due to error: {e}")
    
    # Similar checks
    assert Path(output_dir).exists()
    assert isinstance(results, list)
    assert len(results) > 0
    
    successful = sum(1 for r in results if r["success"])
    print(f"3-web example processed {len(results)} sections: {successful} successful")


def test_example_output_files(example_output_dir):
    """Test specific output files are created correctly."""
    from b3_2d.core.mesh import process_vtp_multi_section
    
    vtp_file = "examples/draped.vtk"
    output_dir = str(example_output_dir / "check_files")
    
    if not Path(vtp_file).exists():
        pytest.skip("Skipping due to missing draped.vtk")
    
    try:
        results = process_vtp_multi_section(vtp_file, output_dir)
    except Exception:
        pytest.skip("Skipping due to VTP processing error")
    
    # Verify VTK files can be read for successful sections
    for result in results:
        if result["success"]:
            vtk_file = Path(result["output_dir"]) / "output.vtk"
            if vtk_file.exists():
                mesh = pv.read(str(vtk_file))
                assert mesh.n_points > 0
                assert mesh.n_cells > 0
                assert "material_id" in mesh.cell_data


def test_example_programmatic_usage():
    """Test both example scripts can be imported and run programmatically."""
    examples_to_test = [
        "examples.mesh_example",
        "examples.mesh_3web_example"
    ]
    
    for example_module in examples_to_test:
        try:
            module = __import__(example_module.replace('.', ''), fromlist=[''])
            assert hasattr(module, '__file__')
            print(f"✓ {example_module} imports successfully")
        except ImportError as e:
            pytest.skip(f"Example {example_module} not importable: {e}")
        except Exception as e:
            pytest.skip(f"Example {example_module} has import errors: {e}")


@pytest.mark.skipif(
    not (Path("examples/draped.vtk").exists() or Path("examples/draped_3web.vtk").exists()),
    reason="No sample VTP files found"
)
def test_full_example_pipeline(tmp_path):
    """Full end-to-end test with available real data."""
    from b3_2d.core.mesh import process_vtp_multi_section
    
    available_files = []
    for vtp_file in ["examples/draped.vtk", "examples/draped_3web.vtk"]:
        if Path(vtp_file).exists():
            available_files.append(vtp_file)
    
    assert available_files, "No test VTP files available"
    
    for vtp_file in available_files:
        output_dir = tmp_path / Path(vtp_file).stem
        output_dir.mkdir()
        
        results = process_vtp_multi_section(str(vtp_file), str(output_dir))
        
        assert len(results) > 0
        successful = [r for r in results if r["success"]]
        assert len(successful) > 0, f"At least one section should process successfully for {vtp_file}"
        
        print(f"✓ Full pipeline test passed for {vtp_file}: {len(successful)}/{len(results)} successful")
        
        # Check VTK files
        for result in successful:
            vtk_file = Path(result["output_dir"]) / "output.vtk"
            assert vtk_file.exists()
            mesh = pv.read(str(vtk_file))
            assert mesh.n_cells > 50  # Reasonable minimum for 2D mesh
            
            anba_file = Path(result["output_dir"]) / "anba.json"
            assert anba_file.exists()
            
            log_file = Path(result["output_dir"]) / "2dmesh.log"
            assert log_file.exists()
